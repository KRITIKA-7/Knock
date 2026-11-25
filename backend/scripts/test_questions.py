import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError
from backend.app.config import settings, split_fetched_detail
from backend.app.schemas import EvidenceReport, GitHubProfile, ResumeProfile, StartupProfile
from backend.app.services.evidence_checker import PROMPT_VERSION as REPORT_PROMPT_VERSION, check_resume
from backend.app.services.fit_scorer import score_fit
from backend.app.services.llm_client import (
    format_cache_summary,
    reset_llm_cache_stats,
    set_llm_cache_enabled,
)
from backend.app.services.question_generator import generate_questions

CACHE_DIR = ROOT / "samples" / ".cache"
STARTUPS_DIR = ROOT / "data" / "startups"


def _report_cache_path(resume: ResumeProfile, github: GitHubProfile) -> Path:
    content_hash = hashlib.sha256(
        f"{resume.model_dump_json()}|{github.model_dump_json()}".encode("utf-8")
    ).hexdigest()
    key = hashlib.sha256(
        f"{content_hash}|{settings.gemini_model}|{REPORT_PROMPT_VERSION}".encode("utf-8")
    ).hexdigest()[:16]
    return CACHE_DIR / f"report_{key}.json"


def _load_or_run_report(
    resume: ResumeProfile,
    github: GitHubProfile,
    no_cache: bool = False,
) -> tuple[EvidenceReport, bool]:
    cache_file = _report_cache_path(resume, github)
    if not no_cache and cache_file.exists():
        try:
            return EvidenceReport.model_validate_json(cache_file.read_text(encoding="utf-8")), True
        except (ValidationError, ValueError):
            pass

    report = check_resume(resume, github)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*")
    parser.add_argument("--resume-json", default=None)
    parser.add_argument("--github-json", default=None)
    parser.add_argument("--startup", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-llm-cache", action="store_true")
    args = parser.parse_args()

    reset_llm_cache_stats()
    set_llm_cache_enabled(not args.no_llm_cache)

    resume_path = args.resume_json
    github_path = args.github_json
    startup_slug = args.startup

    if len(args.args) == 3:
        resume_path, github_path, startup_slug = args.args
    elif len(args.args) == 1 and startup_slug is None:
        startup_slug = args.args[0]
    elif args.args:
        raise ValueError("Pass <resume_json> <github_json> <startup_slug> or use --resume-json/--github-json/--startup")

    if not resume_path or not github_path or not startup_slug:
        raise ValueError("Missing required resume JSON, github JSON, or startup slug")

    resume = ResumeProfile.model_validate_json(
        Path(resume_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    github = GitHubProfile.model_validate_json(
        Path(github_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    startup_file = STARTUPS_DIR / f"{startup_slug}.json"
    if not startup_file.exists():
        raise FileNotFoundError(f"Startup profile not found: {startup_file}")
    startup = StartupProfile.model_validate_json(startup_file.read_text(encoding="utf-8"))

    report, from_cache = _load_or_run_report(resume, github, no_cache=args.no_cache)
    fit_result = score_fit(report, startup)
    qset = generate_questions(fit_result, startup, report=report, github=github)
    total_llm_calls = fit_result.llm_calls + qset.llm_calls

    problem_map = {p.id: p for p in startup.hard_problems}
    print(f"Startup: {startup.name} ({startup.slug}) | Domain: {startup.domain}")
    print(
        f"Report source: {'cache' if from_cache else 'fresh'} | Fit: overall={fit_result.overall:.4f}, domain={fit_result.domain_fit:.4f}, skill={fit_result.skill_fit:.4f}, weak_match={fit_result.weak_match}"
    )
    print(
        f"Models: generator={qset.model}, verifier={qset.verifier_model} | Prompt: {qset.prompt_version}"
    )
    print(
        f"llm_calls: fit={fit_result.llm_calls}, questions={qset.llm_calls}, total={total_llm_calls} | fetch_count={qset.fetch_count} (cached={qset.cached_fetches})"
    )
    print(
        f"Candidates: generated={qset.candidates_generated}, verified={qset.questions_verified}, dropped={qset.questions_dropped}"
    )
    print(format_cache_summary(CACHE_DIR))

    if qset.dropped_questions:
        print(f"Dropped Questions ({len(qset.dropped_questions)}):")
        for dq in qset.dropped_questions:
            files_str = ", ".join(dq.cited_files) if dq.cited_files else "-"
            print(
                f"  - [{dq.id}] {dq.project_name} -> {dq.startup_problem_id} | Verdict: {dq.verdict} | File(s): {files_str}"
            )
            print(f"    Text: {dq.text}")
            print(f"    Reason: {dq.reason}")
    elif qset.dropped_verdict_reasons:
        print("Dropped Verdict Reasons:")
        for r in qset.dropped_verdict_reasons:
            print(f"  - {r}")

    if qset.weak_match:
        print("=" * 70)
        print("WEAK MATCH: True (questions=[])")
        print(f"Reason: {qset.reason}")
        print("Closest Verified Work:")
        if qset.closest_verified_work:
            for item in qset.closest_verified_work:
                print(f"  - {item}")
        else:
            print("  - None")
        print("=" * 70)
        print(qset.model_dump_json(indent=2))
        return

    for q in qset.questions:
        prob = problem_map.get(q.startup_problem_id)
        prob_title = prob.title if prob else q.startup_problem_id
        cited_paths = []
        for ref in q.grounded_in:
            split = split_fetched_detail(ref.detail)
            cited_paths.append(split[0] if split else ref.detail)
        print("=" * 70)
        print(
            f"[{q.id}] Project: {q.project_name} | Time: {q.time_limit_minutes}m | Context length: {q.context_chars} chars"
        )
        print(f"Startup Problem ({q.startup_problem_id}): {prob_title}")
        print(f"Cited File Path(s): {', '.join(cited_paths)}")
        print(f"Context (first 300 chars):\n{q.context_preview[:300]}")
        if q.verification_verdict:
            print(f"Verifier ({q.verification_verdict}): {q.verification_reason}")
        print(f"Question:\n{q.text}")
        print("Grounded In:")
        for ref in q.grounded_in:
            print(f"  - [{ref.kind}] (independent={ref.independent}) {ref.detail}")
    print("=" * 70)
    print(qset.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
