import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import (
    Candidate,
    EvidenceReportRecord,
    FitResultRecord,
    GitHubProfileRecord,
    ResumeProfileRecord,
    TestSessionRecord,
    UnlockRecord,
    utcnow,
)
from backend.app.schemas import CandidateCreate, ConsentUpdate, EvidenceReport, FitResult
from backend.app.services.pipeline import (
    GitHubUserNotFoundError,
    InvalidGitHubUsernameError,
    build_rejection_feedback,
    delete_candidate_cache,
    load_all_startups,
    run_analysis,
    verify_github_username,
)

router = APIRouter(prefix="/candidates", tags=["candidates"])


def _get_candidate_or_404(db: Session, candidate_id: str) -> Candidate:
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")
    return candidate


@router.post("", status_code=201)
def create_candidate(payload: CandidateCreate, db: Session = Depends(get_db)):
    now = utcnow()
    candidate = Candidate(
        github_username=payload.github_username.strip().lstrip("@"),
        processing_notice_accepted=payload.processing_notice_accepted,
        processing_notice_accepted_at=now if payload.processing_notice_accepted else None,
        data_consent=payload.data_consent,
        consent_at=now if payload.data_consent else None,
        consent_withdrawn_at=None,
        created_at=now,
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return {
        "id": candidate.id,
        "github_username": candidate.github_username,
        "processing_notice_accepted": candidate.processing_notice_accepted,
        "processing_notice_accepted_at": (
            candidate.processing_notice_accepted_at.isoformat()
            if candidate.processing_notice_accepted_at
            else None
        ),
        "data_consent": candidate.data_consent,
        "consent_at": candidate.consent_at.isoformat() if candidate.consent_at else None,
        "consent_withdrawn_at": None,
        "analysis_status": candidate.analysis_status or "not_started",
        "created_at": candidate.created_at.isoformat(),
    }


@router.patch("/{candidate_id}/consent")
def update_consent(candidate_id: str, payload: ConsentUpdate, db: Session = Depends(get_db)):
    candidate = _get_candidate_or_404(db, candidate_id)
    now = utcnow()
    candidate.data_consent = payload.data_consent
    if payload.data_consent:
        candidate.consent_at = now
        candidate.consent_withdrawn_at = None
    else:
        candidate.consent_withdrawn_at = now

    if payload.processing_notice_accepted is not None:
        candidate.processing_notice_accepted = payload.processing_notice_accepted
        if payload.processing_notice_accepted and candidate.processing_notice_accepted_at is None:
            candidate.processing_notice_accepted_at = now
        elif not payload.processing_notice_accepted:
            candidate.processing_notice_accepted_at = None

    db.commit()
    db.refresh(candidate)
    return {
        "candidate_id": candidate.id,
        "processing_notice_accepted": candidate.processing_notice_accepted,
        "processing_notice_accepted_at": (
            candidate.processing_notice_accepted_at.isoformat()
            if candidate.processing_notice_accepted_at
            else None
        ),
        "data_consent": candidate.data_consent,
        "consent_at": candidate.consent_at.isoformat() if candidate.consent_at else None,
        "consent_withdrawn_at": (
            candidate.consent_withdrawn_at.isoformat()
            if candidate.consent_withdrawn_at
            else None
        ),
    }


@router.post("/{candidate_id}/analyze", status_code=202)
async def analyze_candidate(
    candidate_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    candidate = _get_candidate_or_404(db, candidate_id)
    if not candidate.processing_notice_accepted:
        raise HTTPException(
            status_code=403,
            detail="Processing notice must be accepted (processing_notice_accepted=true) before analysis.",
        )

    if candidate.analysis_status in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Analysis already {candidate.analysis_status} for candidate {candidate_id}.",
        )

    try:
        clean_username = verify_github_username(candidate.github_username)
    except InvalidGitHubUsernameError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except GitHubUserNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except httpx.HTTPError as err:
        raise HTTPException(
            status_code=503,
            detail="GitHub verification service is temporarily unavailable. Please retry shortly.",
            headers={"Retry-After": "15"},
        ) from err

    filename = (file.filename or "").lower()
    if filename and not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    max_bytes = settings.max_resume_pdf_bytes
    pdf_bytes = await file.read(max_bytes + 1)
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Resume PDF exceeds maximum size of {max_bytes // (1024 * 1024)}MB.",
        )
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")

    candidate.github_username = clean_username
    candidate.analysis_status = "queued"
    candidate.failing_stage = None
    candidate.analysis_error = None
    candidate.analysis_updated_at = utcnow()
    db.commit()

    background_tasks.add_task(run_analysis, candidate.id, pdf_bytes)
    return {
        "candidate_id": candidate.id,
        "status": "queued",
    }


@router.get("/{candidate_id}/analysis")
def get_analysis(candidate_id: str, db: Session = Depends(get_db)):
    candidate = _get_candidate_or_404(db, candidate_id)
    report_rec = (
        db.query(EvidenceReportRecord)
        .filter(EvidenceReportRecord.candidate_id == candidate_id)
        .order_by(EvidenceReportRecord.id.desc())
        .first()
    )

    if report_rec is None:
        return {
            "candidate_id": candidate.id,
            "github_username": candidate.github_username,
            "status": candidate.analysis_status or "not_started",
            "failing_stage": candidate.failing_stage,
            "error": candidate.analysis_error,
            "evidence_gate_passed": None,
            "evidence_report": None,
            "rejection_feedback": [],
            "all_gaps": [],
        }

    report = EvidenceReport.model_validate(report_rec.report_data)
    report_payload = report.model_dump()
    report_payload.pop("model", None)
    for item in report_payload.get("results", []):
        if isinstance(item, dict):
            item.pop("model", None)
    prioritized_feedback, all_gaps = build_rejection_feedback(report)
    return {
        "candidate_id": candidate.id,
        "github_username": candidate.github_username,
        "status": candidate.analysis_status or "done",
        "failing_stage": candidate.failing_stage,
        "error": candidate.analysis_error,
        "evidence_gate_passed": report_rec.evidence_gate_passed,
        "evidence_report": report_payload,
        "rejection_feedback": prioritized_feedback,
        "all_gaps": all_gaps,
        "prompt_version": report_rec.prompt_version,
        "created_at": report_rec.created_at.isoformat(),
    }


@router.get("/{candidate_id}/startups")
def get_candidate_startups(candidate_id: str, db: Session = Depends(get_db)):
    candidate = _get_candidate_or_404(db, candidate_id)
    records = (
        db.query(FitResultRecord)
        .filter(FitResultRecord.candidate_id == candidate_id)
        .order_by(FitResultRecord.id.desc())
        .all()
    )

    if not records:
        if candidate.analysis_status in ("queued", "running"):
            raise HTTPException(status_code=409, detail="Analysis is still running.")
        if candidate.analysis_status == "failed":
            raise HTTPException(
                status_code=503 if "unavailable" in (candidate.analysis_error or "").lower() else 400,
                detail=candidate.analysis_error or f"Analysis failed at stage '{candidate.failing_stage}'. Please retry.",
            )
        raise HTTPException(status_code=404, detail="No startup fit results available yet.")

    startup_meta = {sp.slug: sp for sp in load_all_startups()}
    latest_by_slug: dict[str, FitResultRecord] = {}
    for rec in records:
        if rec.startup_slug not in latest_by_slug:
            latest_by_slug[rec.startup_slug] = rec

    ordered = sorted(
        latest_by_slug.values(),
        key=lambda r: (-r.overall, -r.domain_fit, -r.skill_fit, r.startup_slug),
    )

    items = []
    for rank, rec in enumerate(ordered, start=1):
        fit = FitResult.model_validate(rec.fit_data)
        meta = startup_meta.get(rec.startup_slug)
        items.append(
            {
                "rank": rank,
                "startup_slug": rec.startup_slug,
                "name": meta.name if meta else rec.startup_slug,
                "domain": meta.domain if meta else "",
                "one_liner": meta.one_liner if meta else "",
                "tech_stack": meta.tech_stack if meta else [],
                "overall": fit.overall,
                "domain_fit": fit.domain_fit,
                "skill_fit": fit.skill_fit,
                "weak_match": fit.weak_match,
                "gaps": fit.gaps,
                "matches": [m.model_dump() for m in fit.matches],
                "tech_overlap_sources": fit.tech_overlap_sources,
                "prompt_version": rec.prompt_version,
            }
        )

    return {
        "candidate_id": candidate.id,
        "status": candidate.analysis_status or "done",
        "startups": items,
    }


@router.delete("/{candidate_id}")
def delete_candidate(candidate_id: str, db: Session = Depends(get_db)):
    candidate = _get_candidate_or_404(db, candidate_id)
    if candidate.analysis_status in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail="Cannot delete candidate while analysis is queued or running.",
        )

    try:
        db.query(UnlockRecord).filter(UnlockRecord.candidate_id == candidate_id).delete()
        db.query(TestSessionRecord).filter(TestSessionRecord.candidate_id == candidate_id).delete()
        db.query(FitResultRecord).filter(FitResultRecord.candidate_id == candidate_id).delete()
        db.query(EvidenceReportRecord).filter(EvidenceReportRecord.candidate_id == candidate_id).delete()
        db.query(GitHubProfileRecord).filter(GitHubProfileRecord.candidate_id == candidate_id).delete()
        db.query(ResumeProfileRecord).filter(ResumeProfileRecord.candidate_id == candidate_id).delete()
        db.delete(candidate)
        db.commit()
    except Exception as err:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete candidate records.") from err

    try:
        delete_candidate_cache(candidate_id)
    except OSError as err:
        raise HTTPException(status_code=500, detail="Failed to delete candidate cached files.") from err

    return {"deleted": True, "candidate_id": candidate_id}
