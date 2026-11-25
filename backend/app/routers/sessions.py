from datetime import datetime, timedelta, timezone
import logging
from fastapi import APIRouter, Depends, HTTPException
from google.genai import errors as genai_errors
import httpx
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import (
    Candidate,
    EvidenceReportRecord,
    FitResultRecord,
    GitHubProfileRecord,
    TestSessionRecord,
    UnlockRecord,
    utcnow,
)
from backend.app.schemas import (
    AnswerSubmit,
    EvidenceReport,
    FitResult,
    GitHubProfile,
    Question,
    SessionCreate,
    TestSession,
)
from backend.app.services.answer_scorer import (
    PROMPT_VERSION as SCORER_PROMPT_VERSION,
    _validate_answer_length,
    evaluate_session,
    generate_followup,
)
from backend.app.services.llm_client import scoped_llm_cache_dir
from backend.app.services.pipeline import candidate_cache_dir, load_startup_profile
from backend.app.services.question_generator import generate_questions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])

_RETRY_DETAIL = "Upstream assessment service is temporarily unavailable. Please retry in a few moments."


def _raise_upstream_503(context: str, err: Exception) -> None:
    logger.warning("%s failed due to %s", context, type(err).__name__)
    raise HTTPException(
        status_code=503,
        detail=_RETRY_DETAIL,
        headers={"Retry-After": "30"},
    ) from err


def _get_session_or_404(db: Session, session_id: str) -> TestSessionRecord:
    rec = db.query(TestSessionRecord).filter(TestSessionRecord.id == session_id).first()
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return rec


def _elapsed_seconds(started_at: dict, key: str) -> float:
    now = datetime.now(timezone.utc)
    raw_ts = started_at.get(key)
    if not isinstance(raw_ts, str):
        return 0.0
    try:
        dt = datetime.fromisoformat(raw_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return round(max(0.0, (now - dt).total_seconds()), 2)
    except ValueError:
        return 0.0


def _score_session_record(rec: TestSessionRecord) -> TestSession:
    questions = [Question.model_validate(q) for q in (rec.questions or [])]
    evaluated = evaluate_session(
        startup_slug=rec.startup_slug,
        questions=questions,
        answers=rec.answers or {},
        followup_answers=rec.followup_answers or {},
        followups=rec.followups or {},
        seconds_per_question=rec.seconds_per_question or {},
        used_dispute=rec.used_dispute,
    )
    rec.scores = [s.model_dump() for s in evaluated.scores]
    rec.total = evaluated.total
    rec.passed = evaluated.passed
    rec.model = evaluated.model
    rec.prompt_version = evaluated.prompt_version
    return evaluated


@router.post("", status_code=201)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == payload.candidate_id).first()
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate {payload.candidate_id} not found")

    startup = load_startup_profile(payload.startup_slug)
    if startup is None:
        raise HTTPException(status_code=404, detail=f"Startup '{payload.startup_slug}' not found")

    fit_rec = (
        db.query(FitResultRecord)
        .filter(
            FitResultRecord.candidate_id == payload.candidate_id,
            FitResultRecord.startup_slug == payload.startup_slug,
        )
        .order_by(FitResultRecord.id.desc())
        .first()
    )
    if fit_rec is None:
        if candidate.analysis_status in ("queued", "running"):
            raise HTTPException(status_code=409, detail="Candidate analysis is still running.")
        raise HTTPException(
            status_code=400,
            detail=f"No fit result for candidate {payload.candidate_id} and startup '{payload.startup_slug}'; run analysis first.",
        )

    rep_rec = (
        db.query(EvidenceReportRecord)
        .filter(EvidenceReportRecord.candidate_id == payload.candidate_id)
        .order_by(EvidenceReportRecord.id.desc())
        .first()
    )
    gh_rec = (
        db.query(GitHubProfileRecord)
        .filter(GitHubProfileRecord.candidate_id == payload.candidate_id)
        .order_by(GitHubProfileRecord.id.desc())
        .first()
    )

    fit = FitResult.model_validate(fit_rec.fit_data)
    report = EvidenceReport.model_validate(rep_rec.report_data) if rep_rec else None
    github = GitHubProfile.model_validate(gh_rec.profile_data) if gh_rec else None

    try:
        with scoped_llm_cache_dir(candidate_cache_dir(candidate.id) / "llm"):
            qset = generate_questions(fit, startup, report=report, github=github)
    except (genai_errors.APIError, httpx.HTTPError, TimeoutError, ValueError) as err:
        _raise_upstream_503("question generation", err)

    now_iso = datetime.now(timezone.utc).isoformat()
    started = {q.id: now_iso for q in qset.questions}

    rec = TestSessionRecord(
        candidate_id=candidate.id,
        startup_slug=startup.slug,
        weak_match=qset.weak_match,
        weak_match_reason=qset.reason,
        closest_verified_work=qset.closest_verified_work,
        questions=[q.model_dump() for q in qset.questions],
        answers={},
        followups={},
        followup_answers={},
        scores=[],
        total=0.0 if qset.weak_match else None,
        passed=False,
        used_dispute=payload.used_dispute,
        seconds_per_question={},
        question_started_at=started,
        model=f"{qset.model} / {qset.verifier_model}" if qset.verifier_model else qset.model,
        prompt_version=qset.prompt_version,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    return {
        "session_id": rec.id,
        "candidate_id": rec.candidate_id,
        "startup_slug": rec.startup_slug,
        "weak_match": qset.weak_match,
        "reason": qset.reason,
        "closest_verified_work": qset.closest_verified_work,
        "questions": [q.model_dump() for q in qset.questions],
        "candidates_generated": qset.candidates_generated,
        "questions_verified": qset.questions_verified,
        "questions_dropped": qset.questions_dropped,
        "prompt_version": rec.prompt_version,
    }


@router.post("/{session_id}/answer")
def submit_answer(session_id: str, payload: AnswerSubmit, db: Session = Depends(get_db)):
    rec = _get_session_or_404(db, session_id)
    if rec.weak_match or not rec.questions:
        raise HTTPException(
            status_code=400,
            detail="Cannot submit answers for a weak_match session with no questions.",
        )

    questions = [Question.model_validate(q) for q in rec.questions]
    target_q = next((q for q in questions if q.id == payload.question_id), None)
    if target_q is None:
        raise HTTPException(status_code=404, detail=f"Question '{payload.question_id}' not found in session {session_id}")

    err_msg = _validate_answer_length(payload.answer)
    if err_msg is not None:
        raise HTTPException(status_code=400, detail=err_msg)

    try:
        with scoped_llm_cache_dir(candidate_cache_dir(rec.candidate_id) / "llm"):
            followup = generate_followup(target_q, payload.answer)
    except (genai_errors.APIError, httpx.HTTPError, TimeoutError, ValueError) as err:
        _raise_upstream_503("followup generation", err)

    started = dict(rec.question_started_at or {})
    elapsed = _elapsed_seconds(started, payload.question_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    started[f"{payload.question_id}_followup"] = now_iso

    rec.question_started_at = started
    rec.seconds_per_question = {**(rec.seconds_per_question or {}), payload.question_id: elapsed}
    rec.answers = {**(rec.answers or {}), payload.question_id: payload.answer}
    rec.followups = {**(rec.followups or {}), payload.question_id: followup}
    db.commit()

    return {
        "session_id": rec.id,
        "question_id": payload.question_id,
        "followup": followup,
    }


@router.post("/{session_id}/followup")
def submit_followup(session_id: str, payload: AnswerSubmit, db: Session = Depends(get_db)):
    rec = _get_session_or_404(db, session_id)
    if rec.weak_match or not rec.questions:
        raise HTTPException(
            status_code=400,
            detail="Cannot submit follow-up for a weak_match session with no questions.",
        )

    questions = [Question.model_validate(q) for q in rec.questions]
    target_q = next((q for q in questions if q.id == payload.question_id), None)
    if target_q is None:
        raise HTTPException(status_code=404, detail=f"Question '{payload.question_id}' not found in session {session_id}")

    answers = dict(rec.answers or {})
    if payload.question_id not in answers:
        raise HTTPException(
            status_code=400,
            detail=f"Initial answer for '{payload.question_id}' must be submitted before follow-up.",
        )

    err_msg = _validate_answer_length(payload.answer)
    if err_msg is not None:
        raise HTTPException(status_code=400, detail=err_msg)

    started = dict(rec.question_started_at or {})
    fu_key = f"{payload.question_id}_followup"
    elapsed = _elapsed_seconds(started, fu_key)
    rec.seconds_per_question = {**(rec.seconds_per_question or {}), fu_key: elapsed}
    rec.followup_answers = {**(rec.followup_answers or {}), payload.question_id: payload.answer}

    try:
        with scoped_llm_cache_dir(candidate_cache_dir(rec.candidate_id) / "llm"):
            evaluated = _score_session_record(rec)
    except (genai_errors.APIError, httpx.HTTPError, TimeoutError, ValueError) as err:
        db.rollback()
        _raise_upstream_503("session scoring", err)

    db.commit()
    q_score = next((s for s in evaluated.scores if s.question_id == payload.question_id), None)
    return {
        "session_id": rec.id,
        "question_id": payload.question_id,
        "score": q_score.score if q_score else 0.0,
        "followup_score": q_score.followup_score if q_score else 0.0,
        "combined_score": q_score.combined_score if q_score else 0.0,
        "total": evaluated.total,
        "passed": evaluated.passed,
    }


@router.get("/{session_id}/result")
def get_session_result(session_id: str, db: Session = Depends(get_db)):
    rec = _get_session_or_404(db, session_id)
    if rec.questions and rec.answers:
        try:
            with scoped_llm_cache_dir(candidate_cache_dir(rec.candidate_id) / "llm"):
                evaluated = _score_session_record(rec)
        except (genai_errors.APIError, httpx.HTTPError, TimeoutError, ValueError) as err:
            db.rollback()
            _raise_upstream_503("session result scoring", err)
        db.commit()
        scores_out = [s.model_dump() for s in evaluated.scores]
        total_out = evaluated.total
        passed_out = evaluated.passed
        prompt_out = evaluated.prompt_version
    else:
        scores_out = rec.scores or []
        total_out = rec.total if rec.total is not None else 0.0
        passed_out = bool(rec.passed)
        prompt_out = rec.prompt_version or SCORER_PROMPT_VERSION

    threshold = settings.min_test_score_with_dispute if rec.used_dispute else settings.min_test_score
    return {
        "session_id": rec.id,
        "candidate_id": rec.candidate_id,
        "startup_slug": rec.startup_slug,
        "weak_match": rec.weak_match,
        "questions": rec.questions or [],
        "answers": rec.answers or {},
        "followups": rec.followups or {},
        "followup_answers": rec.followup_answers or {},
        "scores": scores_out,
        "total": total_out,
        "passed": passed_out,
        "used_dispute": rec.used_dispute,
        "required_score": threshold,
        "seconds_per_question": rec.seconds_per_question or {},
        "prompt_version": prompt_out,
    }


@router.post("/{session_id}/unlock")
def unlock_founder(session_id: str, db: Session = Depends(get_db)):
    rec = _get_session_or_404(db, session_id)
    candidate = db.query(Candidate).filter(Candidate.id == rec.candidate_id).first()
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate {rec.candidate_id} not found")

    report_rec = (
        db.query(EvidenceReportRecord)
        .filter(EvidenceReportRecord.candidate_id == candidate.id)
        .order_by(EvidenceReportRecord.id.desc())
        .first()
    )
    if report_rec is None or not report_rec.evidence_gate_passed:
        raise HTTPException(
            status_code=403,
            detail="Unlock denied: candidate has not passed the evidence gate.",
        )

    if rec.weak_match or not rec.questions or not rec.answers:
        raise HTTPException(
            status_code=403,
            detail="Unlock denied: session has not been completed and passed.",
        )

    try:
        with scoped_llm_cache_dir(candidate_cache_dir(candidate.id) / "llm"):
            evaluated = _score_session_record(rec)
    except (genai_errors.APIError, httpx.HTTPError, TimeoutError, ValueError) as err:
        db.rollback()
        _raise_upstream_503("unlock verification", err)
    db.commit()
    if not evaluated.passed:
        threshold = settings.min_test_score_with_dispute if rec.used_dispute else settings.min_test_score
        raise HTTPException(
            status_code=403,
            detail=f"Unlock denied: session score {evaluated.total:.4f} is below required {threshold:.2f}.",
        )

    cutoff = utcnow() - timedelta(days=7)
    cand_week_count = (
        db.query(UnlockRecord)
        .filter(UnlockRecord.candidate_id == candidate.id, UnlockRecord.created_at >= cutoff)
        .count()
    )
    if cand_week_count >= settings.max_unlocks_per_week:
        raise HTTPException(
            status_code=429,
            detail=f"Unlock limit reached: candidate already unlocked {cand_week_count}/{settings.max_unlocks_per_week} startups in the last 7 days.",
        )

    startup_week_count = (
        db.query(UnlockRecord)
        .filter(UnlockRecord.startup_slug == rec.startup_slug, UnlockRecord.created_at >= cutoff)
        .count()
    )
    if startup_week_count >= settings.max_unlocks_per_startup_per_week:
        raise HTTPException(
            status_code=429,
            detail=f"Startup unlock limit reached: '{rec.startup_slug}' reached {startup_week_count}/{settings.max_unlocks_per_startup_per_week} unlocks in the last 7 days.",
        )

    startup = load_startup_profile(rec.startup_slug)
    if startup is None:
        raise HTTPException(status_code=404, detail=f"Startup '{rec.startup_slug}' not found")

    unlock = UnlockRecord(
        candidate_id=candidate.id,
        startup_slug=rec.startup_slug,
    )
    db.add(unlock)
    db.commit()
    db.refresh(unlock)

    return {
        "unlocked": True,
        "candidate_id": candidate.id,
        "startup_slug": rec.startup_slug,
        "founder": startup.founder.model_dump(),
        "unlocked_at": unlock.created_at.isoformat(),
    }
