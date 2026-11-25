import hashlib
import logging
from pathlib import Path
import re
import shutil
import tempfile

from google.genai import errors as genai_errors
import httpx
from pydantic import ValidationError
from sqlalchemy.orm import Session

from backend.app.config import BASE_DIR, is_production_env, settings
from backend.app.database import SessionLocal
from backend.app.models import (
    Candidate,
    EvidenceReportRecord,
    FitResultRecord,
    GitHubProfileRecord,
    ResumeProfileRecord,
    TestSessionRecord,
    utcnow,
)
from backend.app.schemas import (
    ClaimResult,
    EvidenceReport,
    GitHubProfile,
    ResumeProfile,
    StartupProfile,
    Verdict,
)
from backend.app.services.evidence_checker import (
    PROMPT_VERSION as EVIDENCE_PROMPT_VERSION,
    check_resume,
)
from backend.app.services.fit_scorer import score_fit
from backend.app.services.github_analyzer import (
    GITHUB_API_BASE,
    _request_with_rate_limit,
    analyze_github,
)
from backend.app.services.llm_client import scoped_llm_cache_dir
from backend.app.services.pdf_extractor import extract_text
from backend.app.services.resume_parser import parse_resume

logger = logging.getLogger(__name__)

STARTUPS_DIR = BASE_DIR / "data" / "startups"
CACHE_DIR = BASE_DIR / "samples" / ".cache"
CANDIDATES_CACHE_DIR = CACHE_DIR / "candidates"
RESUME_PROMPT_VERSION = "v1"
GITHUB_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_CLAIM_TYPE_LABELS = {
    "feature": "feature",
    "technology": "technology",
    "numeric_metric": "metric",
    "deployment": "deployment",
    "timeline": "timeline",
    "other": "project",
}
_CLAIM_PRIORITY = {
    "feature": 0,
    "numeric_metric": 1,
    "deployment": 2,
    "technology": 3,
    "timeline": 4,
    "other": 5,
}
_ACTION_PRIORITY = {
    "link_repo": 0,
    "confirm_email": 1,
    "add_output_files": 2,
}


class InvalidGitHubUsernameError(ValueError):
    pass


class GitHubUserNotFoundError(LookupError):
    pass


def verify_github_username(username: str) -> str:
    clean = username.strip().lstrip("@")
    if not clean or not GITHUB_USERNAME_PATTERN.match(clean):
        raise InvalidGitHubUsernameError(f"Invalid GitHub username: {username!r}")

    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        resp = _request_with_rate_limit(client, f"{GITHUB_API_BASE}/users/{clean}")
    if resp.status_code == 404:
        raise GitHubUserNotFoundError(f"GitHub user not found: {clean}")
    resp.raise_for_status()
    return clean


def candidate_cache_dir(candidate_id: str) -> Path:
    clean = str(candidate_id).strip()
    if not SAFE_ID_PATTERN.match(clean):
        raise ValueError("Invalid candidate_id for cache path")
    return CANDIDATES_CACHE_DIR / clean


def delete_candidate_cache(candidate_id: str) -> None:
    cdir = candidate_cache_dir(candidate_id)
    if cdir.exists():
        shutil.rmtree(cdir)


def load_startup_profile(slug: str) -> StartupProfile | None:
    path = STARTUPS_DIR / f"{slug}.json"
    if not path.exists():
        return None
    return StartupProfile.model_validate_json(path.read_text(encoding="utf-8"))


def load_all_startups() -> list[StartupProfile]:
    files = sorted(STARTUPS_DIR.glob("*.json"))
    return [StartupProfile.model_validate_json(p.read_text(encoding="utf-8")) for p in files]


def _collect_resume_links(profile: ResumeProfile) -> list[str]:
    links: list[str] = []
    for exp in profile.experience:
        links.extend(exp.links)
    for proj in profile.projects:
        links.extend(proj.links)
    return links


def _parsed_cache_path(raw_text: str) -> Path:
    text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"parsed_{text_hash}_{settings.gemini_model}_{RESUME_PROMPT_VERSION}.json"


def _github_cache_path(username: str, links: list[str]) -> Path:
    key_material = f"{username.lower()}|{'|'.join(sorted(links))}|"
    key_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"github_{username.lower()}_{key_hash}.json"


def _report_cache_path(resume: ResumeProfile, github: GitHubProfile) -> Path:
    content_hash = hashlib.sha256(
        f"{resume.model_dump_json()}|{github.model_dump_json()}".encode("utf-8")
    ).hexdigest()
    key = hashlib.sha256(
        f"{content_hash}|{settings.gemini_model}|{EVIDENCE_PROMPT_VERSION}".encode("utf-8")
    ).hexdigest()[:16]
    return CACHE_DIR / f"report_{key}.json"


def _clear_prior_analysis_rows(db: Session, candidate_id: str) -> None:
    db.query(FitResultRecord).filter(FitResultRecord.candidate_id == candidate_id).delete()
    db.query(EvidenceReportRecord).filter(EvidenceReportRecord.candidate_id == candidate_id).delete()
    db.query(GitHubProfileRecord).filter(GitHubProfileRecord.candidate_id == candidate_id).delete()
    db.query(ResumeProfileRecord).filter(ResumeProfileRecord.candidate_id == candidate_id).delete()
    db.commit()


def _extract_from_bytes(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        return extract_text(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _classify_group_action(
    project_name: str,
    claim_type: str,
    items: list[ClaimResult],
) -> tuple[str, str, str]:
    count = len(items)
    label = _CLAIM_TYPE_LABELS.get(claim_type, claim_type)
    combined_notes = " ".join(
        f"{' '.join(c.signals)} {c.reason} {c.candidate_action or ''}" for c in items
    ).lower()

    if (
        "unlinked commits" in combined_notes
        or "confirm identity" in combined_notes
        or "confirm commit email" in combined_notes
    ):
        return (
            "confirm_email",
            f"{count} {label} claim(s) in '{project_name}' match commits from an unlinked Git email.",
            f"Confirm your commit email address for '{project_name}' so those commits attribute to your GitHub account.",
        )

    suggested = sorted({c.suggested_repo for c in items if c.suggested_repo})
    if (
        suggested
        or any(c.matched_repo is None for c in items)
        or "no linked public repo" in combined_notes
        or "repo that is not linked" in combined_notes
        or "link a public repo" in combined_notes
    ):
        fix = (
            f"Link the repository on your resume for '{project_name}' (suggested: {', '.join(suggested[:2])})."
            if suggested
            else f"Link a public GitHub repository for '{project_name}' on your resume and re-run analysis."
        )
        return (
            "link_repo",
            f"{count} {label} claim(s) in '{project_name}' do not point to a linked public repository.",
            fix,
        )

    if (
        claim_type in ("numeric_metric", "deployment")
        or "numbers not found" in combined_notes
        or "deploy config" in combined_notes
        or "live homepage" in combined_notes
        or "benchmark" in combined_notes
        or "output" in combined_notes
    ):
        if claim_type == "numeric_metric":
            return (
                "add_output_files",
                f"{count} metric claim(s) in '{project_name}' could not be verified from repository files or README.",
                f"Add output files, benchmark logs, or evaluation artifacts documenting these numbers in '{project_name}'.",
            )
        return (
            "add_output_files",
            f"{count} {label} claim(s) in '{project_name}' lacked verifiable config or output artifacts in the repository.",
            f"Add output files, deployment config, or README verification artifacts in '{project_name}'.",
        )

    return (
        "add_output_files",
        f"{count} {label} claim(s) in '{project_name}' lacked verifiable code or output evidence in the repository.",
        f"Add supporting code, tests, benchmark/output files, or README details in '{project_name}', or remove unverifiable non-public claims from your resume and re-upload.",
    )


def build_rejection_feedback(report: EvidenceReport) -> tuple[list[dict], list[dict]]:
    project_map = {p.project_name: p for p in report.projects}
    grouped: dict[tuple[str, str], list[ClaimResult]] = {}

    for r in report.results:
        if r.verdict in (
            Verdict.INSUFFICIENT_EVIDENCE,
            Verdict.UNSUPPORTED,
            Verdict.CONTRADICTED,
        ):
            grouped.setdefault((r.project_name, r.claim_type), []).append(r)

    entries: list[dict] = []
    for (project_name, claim_type), items in grouped.items():
        proj = project_map.get(project_name)
        blocking_gate = bool(proj is not None and not proj.evidenced)
        action_type, summary, how_to_fix = _classify_group_action(project_name, claim_type, items)
        entries.append(
            {
                "project_name": project_name,
                "claim_type": claim_type,
                "action_type": action_type,
                "blocking_gate": blocking_gate,
                "count": len(items),
                "summary": summary,
                "how_to_fix": how_to_fix,
                "claims": [c.claim_text for c in items],
            }
        )

    for proj in report.projects:
        if not proj.evidenced and (proj.project_name, "feature") not in grouped:
            if proj.repo is None:
                action_type = "link_repo"
                how_to_fix = f"Link a public GitHub repository for '{proj.project_name}' on your resume."
            else:
                action_type = "add_output_files"
                how_to_fix = f"Add supporting code, tests, benchmark/output files, or README details showing core features in '{proj.project_name}'."
            entries.append(
                {
                    "project_name": proj.project_name,
                    "claim_type": "feature",
                    "action_type": action_type,
                    "blocking_gate": True,
                    "count": 1,
                    "summary": f"Project '{proj.project_name}' did not pass the evidence gate ({proj.evidenced_reason}).",
                    "how_to_fix": how_to_fix,
                    "claims": [],
                }
            )

    entries.sort(
        key=lambda e: (
            0 if e["blocking_gate"] else 1,
            _CLAIM_PRIORITY.get(e["claim_type"], 9),
            _ACTION_PRIORITY.get(e["action_type"], 9),
            -e["count"],
            e["project_name"],
            e["claim_type"],
        )
    )
    return entries[:5], entries


def export_anonymized_candidates(db: Session) -> list[dict]:
    consented = (
        db.query(Candidate)
        .filter(
            Candidate.data_consent.is_(True),
            Candidate.consent_withdrawn_at.is_(None),
        )
        .all()
    )
    records: list[dict] = []
    for cand in consented:
        rep = (
            db.query(EvidenceReportRecord)
            .filter(EvidenceReportRecord.candidate_id == cand.id)
            .order_by(EvidenceReportRecord.id.desc())
            .first()
        )
        fits = (
            db.query(FitResultRecord)
            .filter(FitResultRecord.candidate_id == cand.id)
            .all()
        )
        sessions = (
            db.query(TestSessionRecord)
            .filter(TestSessionRecord.candidate_id == cand.id)
            .all()
        )
        records.append(
            {
                "evidence_gate_passed": rep.evidence_gate_passed if rep else None,
                "fit_scores": [
                    {
                        "startup_slug": f.startup_slug,
                        "overall": f.overall,
                        "domain_fit": f.domain_fit,
                        "skill_fit": f.skill_fit,
                        "weak_match": f.weak_match,
                    }
                    for f in fits
                ],
                "session_outcomes": [
                    {
                        "startup_slug": s.startup_slug,
                        "weak_match": s.weak_match,
                        "total": s.total,
                        "passed": s.passed,
                    }
                    for s in sessions
                ],
            }
        )
    return records


def _sanitize_pipeline_error(err: Exception, stage: str) -> str:
    if isinstance(err, genai_errors.APIError):
        return "Upstream AI service is temporarily unavailable. Please retry in a few moments."
    if isinstance(err, (httpx.HTTPError, TimeoutError)):
        return "Upstream repository lookup failed or timed out. Please retry shortly."
    if stage == "extract":
        return "Could not extract readable text from the uploaded PDF."
    return f"Analysis could not complete during the '{stage}' stage. Please retry shortly."


def mark_interrupted_analyses(session_factory=SessionLocal) -> int:
    db: Session = session_factory()
    try:
        stuck = (
            db.query(Candidate)
            .filter(Candidate.analysis_status.in_(("queued", "running")))
            .all()
        )
        if not stuck:
            return 0
        now = utcnow()
        for cand in stuck:
            cand.analysis_status = "failed"
            cand.failing_stage = "interrupted"
            cand.analysis_error = "Analysis was interrupted during server restart. Please retry uploading your resume."
            cand.analysis_updated_at = now
        db.commit()
        return len(stuck)
    finally:
        db.close()


def run_analysis(
    candidate_id: str,
    pdf_bytes: bytes,
    session_factory=SessionLocal,
    use_stage_cache: bool = True,
) -> None:
    db: Session = session_factory()
    stage = "extract"
    try:
        candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
        if candidate is None:
            return

        candidate.analysis_status = "running"
        candidate.failing_stage = None
        candidate.analysis_error = None
        candidate.analysis_updated_at = utcnow()
        db.commit()

        _clear_prior_analysis_rows(db, candidate_id)
        cdir = candidate_cache_dir(candidate_id)
        cdir.mkdir(parents=True, exist_ok=True)
        cand_llm_dir = cdir / "llm"
        allow_shared_read = use_stage_cache and not is_production_env()

        stage = "extract"
        raw_text = _extract_from_bytes(pdf_bytes)

        with scoped_llm_cache_dir(cand_llm_dir):
            stage = "parse"
            parsed_cache = _parsed_cache_path(raw_text)
            resume: ResumeProfile | None = None
            if allow_shared_read and parsed_cache.exists():
                try:
                    resume = ResumeProfile.model_validate_json(parsed_cache.read_text(encoding="utf-8"))
                except (ValidationError, ValueError):
                    resume = None
            if resume is None:
                resume = parse_resume(raw_text)

            (cdir / "resume.json").write_text(resume.model_dump_json(indent=2), encoding="utf-8")
            db.add(
                ResumeProfileRecord(
                    candidate_id=candidate_id,
                    raw_text=raw_text,
                    parsed_profile=resume.model_dump(),
                    model=settings.gemini_model,
                    prompt_version=RESUME_PROMPT_VERSION,
                )
            )
            db.commit()

            stage = "github"
            links = _collect_resume_links(resume)
            gh_cache = _github_cache_path(candidate.github_username, links)
            github: GitHubProfile | None = None
            if allow_shared_read and gh_cache.exists():
                try:
                    github = GitHubProfile.model_validate_json(gh_cache.read_text(encoding="utf-8"))
                except (ValidationError, ValueError):
                    github = None
            if github is None:
                github = analyze_github(candidate.github_username, links)

            (cdir / "github.json").write_text(github.model_dump_json(indent=2), encoding="utf-8")
            db.add(
                GitHubProfileRecord(
                    candidate_id=candidate_id,
                    username=github.username,
                    profile_data=github.model_dump(),
                )
            )
            db.commit()

            stage = "evidence"
            rep_cache = _report_cache_path(resume, github)
            report: EvidenceReport | None = None
            if allow_shared_read and rep_cache.exists():
                try:
                    report = EvidenceReport.model_validate_json(rep_cache.read_text(encoding="utf-8"))
                except (ValidationError, ValueError):
                    report = None
            if report is None:
                report = check_resume(resume, github)

            (cdir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            db.add(
                EvidenceReportRecord(
                    candidate_id=candidate_id,
                    report_data=report.model_dump(),
                    evidence_gate_passed=report.evidence_gate_passed,
                    model=report.model,
                    prompt_version=report.prompt_version,
                )
            )
            db.commit()

            stage = "fit"
            for startup in load_all_startups():
                fit = score_fit(report, startup)
                (cdir / f"fit_{startup.slug}.json").write_text(
                    fit.model_dump_json(indent=2), encoding="utf-8"
                )
                db.add(
                    FitResultRecord(
                        candidate_id=candidate_id,
                        startup_slug=startup.slug,
                        domain_fit=fit.domain_fit,
                        skill_fit=fit.skill_fit,
                        overall=fit.overall,
                        weak_match=fit.weak_match,
                        fit_data=fit.model_dump(),
                        model=fit.model,
                        prompt_version=fit.prompt_version,
                    )
                )
            db.commit()

        candidate.analysis_status = "done"
        candidate.failing_stage = None
        candidate.analysis_error = None
        candidate.analysis_updated_at = utcnow()
        db.commit()
    except Exception as err:
        db.rollback()
        logger.warning("Analysis failed for candidate_id=%s at stage=%s: %s", candidate_id, stage, type(err).__name__)
        candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
        if candidate is not None:
            candidate.analysis_status = "failed"
            candidate.failing_stage = stage
            candidate.analysis_error = _sanitize_pipeline_error(err, stage)
            candidate.analysis_updated_at = utcnow()
            db.commit()
    finally:
        db.close()
