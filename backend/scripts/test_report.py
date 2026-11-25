import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError
from backend.app.config import settings
from backend.app.schemas import Claim, EvidenceRef, EvidenceReport, GitHubProfile, RepoEvidence, ResumeProfile, Verdict
from backend.app.services.evidence_checker import PROMPT_VERSION, check_resume
from backend.app.services.github_analyzer import analyze_github, fetch_file
from backend.app.services.llm_client import (
    format_cache_summary,
    reset_llm_cache_stats,
    set_llm_cache_enabled,
)
from backend.app.services.pdf_extractor import extract_text
from backend.app.services.resume_parser import parse_resume

CACHE_DIR = ROOT / "samples" / ".cache"


def _resume_cache_path(pdf_path: Path) -> Path:
    resolved = str(pdf_path.expanduser().resolve())
    path_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{path_hash}.json"


def _github_cache_path(username: str, links: list[str], confirmed_emails: list[str]) -> Path:
    norm_emails = sorted({e.strip().lower() for e in confirmed_emails if e.strip()})
    key_material = f"{username.lower()}|{'|'.join(sorted(links))}|{'|'.join(norm_emails)}"
    key_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"github_{username.lower()}_{key_hash}.json"


def _report_cache_path(resume: ResumeProfile, github: GitHubProfile) -> Path:
    content_hash = hashlib.sha256(
        f"{resume.model_dump_json()}|{github.model_dump_json()}".encode("utf-8")
    ).hexdigest()
    key = hashlib.sha256(
        f"{content_hash}|{settings.gemini_model}|{PROMPT_VERSION}".encode("utf-8")
    ).hexdigest()[:16]
    return CACHE_DIR / f"report_{key}.json"


def _collect_resume_links(profile: ResumeProfile) -> list[str]:
    links: list[str] = []
    for exp in profile.experience:
        links.extend(exp.links)
    for proj in profile.projects:
        links.extend(proj.links)
    return links


def _load_or_parse_resume(pdf_path: str | None, resume_json: str | None, no_cache: bool) -> ResumeProfile:
    if resume_json:
        return ResumeProfile.model_validate_json(
            Path(resume_json).expanduser().resolve().read_text(encoding="utf-8")
        )
    if not pdf_path:
        raise ValueError("Provide a PDF path or --resume-json")

    path = Path(pdf_path)
    cache_file = _resume_cache_path(path)
    if not no_cache and cache_file.exists():
        try:
            return ResumeProfile.model_validate_json(cache_file.read_text(encoding="utf-8"))
        except (ValidationError, ValueError):
            pass

    text = extract_text(path)
    profile = parse_resume(text)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


def _load_or_analyze_github(
    username: str | None,
    github_json: str | None,
    resume: ResumeProfile,
    confirmed_emails: list[str],
    no_cache: bool,
) -> GitHubProfile:
    if github_json:
        return GitHubProfile.model_validate_json(
            Path(github_json).expanduser().resolve().read_text(encoding="utf-8")
        )
    if not username:
        raise ValueError("Provide a GitHub username or --github-json")

    links = _collect_resume_links(resume)
    cache_file = _github_cache_path(username, links, confirmed_emails)
    if not no_cache and cache_file.exists():
        try:
            return GitHubProfile.model_validate_json(cache_file.read_text(encoding="utf-8"))
        except (ValidationError, ValueError):
            pass

    profile = analyze_github(username, links, confirmed_emails=confirmed_emails)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


def _inject_claims(resume: ResumeProfile, specs: list[str]) -> ResumeProfile:
    if not specs:
        return resume
    updated = resume.model_copy(deep=True)
    for spec in specs:
        parts = [p.strip() for p in spec.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid --inject-claim format (expected '<project>|<claim text>'): {spec}")
        proj_query = parts[0].lower()
        claim_text = parts[1]
        claim_type = parts[2] if len(parts) >= 3 and parts[2] in (
            "technology", "numeric_metric", "feature", "deployment", "timeline", "other"
        ) else "feature"

        target_proj = None
        for proj in updated.projects:
            if proj.name.lower() == proj_query:
                target_proj = proj
                break
        if target_proj is None:
            for proj in updated.projects:
                if proj_query in proj.name.lower() or any(proj_query in link.lower() for link in proj.links):
                    target_proj = proj
                    break
        if target_proj is None:
            raise ValueError(f"Project not found for --inject-claim: {parts[0]}")

        target_proj.claims.append(
            Claim(text=claim_text, type=claim_type, source_excerpt=claim_text)
        )
    return updated


def _snippet_around(source_text: str, query: str, radius: int = 120) -> str:
    cleaned = re.sub(r"^(readme\s*:?\s*)", "", query, flags=re.IGNORECASE).strip(" \"'`")
    idx = source_text.lower().find(cleaned.lower())
    if idx != -1:
        start = max(0, idx - radius)
        end = min(len(source_text), idx + len(cleaned) + radius)
        return source_text[start:end].strip()

    for word in re.findall(r"[A-Za-z0-9_.\-]{4,}", cleaned):
        w_idx = source_text.lower().find(word.lower())
        if w_idx != -1:
            start = max(0, w_idx - radius)
            end = min(len(source_text), w_idx + len(word) + radius)
            return source_text[start:end].strip()
    return source_text[: radius * 2].strip()


def _ref_path_label(ref: EvidenceRef, repo: RepoEvidence | None) -> str:
    if ref.kind == "fetched_file":
        if repo:
            for p in repo.file_tree:
                if p in ref.detail:
                    return p
        return ref.detail.split(":", 1)[0].strip()
    if ref.kind == "file":
        if repo:
            for p in repo.file_tree:
                if p in ref.detail:
                    return p
        return ref.detail.strip()
    if ref.kind == "readme":
        return "README.md"
    if ref.kind == "homepage":
        return (repo.homepage if repo else None) or "homepage"
    if ref.kind == "commit_stat":
        return "git_commits"
    return ref.kind


def _resolve_ref_snippet(ref: EvidenceRef, repo: RepoEvidence | None, fetched_cache: dict[str, str | None]) -> str:
    if repo is None:
        return "(no matched repo)"
    if ref.kind == "dependency":
        matches = [d for d in repo.dependencies if d.lower() in ref.detail.lower()]
        return f"dependencies={matches or repo.dependencies[:10]}"
    if ref.kind == "language":
        matches = {k: v for k, v in repo.languages.items() if k.lower() in ref.detail.lower()}
        return f"languages={matches or repo.languages}"
    if ref.kind == "file":
        matches = [p for p in repo.file_tree if p in ref.detail]
        return f"file_tree match={matches or '(not found)'}"
    if ref.kind == "homepage":
        return f"homepage={repo.homepage}"
    if ref.kind == "commit_stat":
        return (
            f"user={repo.user_commit_count}, unlinked={repo.unlinked_commit_count}, "
            f"other={repo.other_author_count}, active_days={repo.active_days}, "
            f"first={repo.first_commit_at}, last={repo.last_commit_at}"
        )
    if ref.kind == "readme":
        if not repo.readme_excerpt:
            return "(empty readme)"
        return _snippet_around(repo.readme_excerpt, ref.detail)
    if ref.kind == "fetched_file":
        path_match = next((p for p in repo.file_tree if p in ref.detail), None)
        if not path_match:
            path_match = ref.detail.split(":", 1)[0].strip()
        cache_key = f"{repo.full_name}:{path_match}"
        if cache_key not in fetched_cache:
            fetched_cache[cache_key] = fetch_file(repo.full_name, path_match, max_bytes=20000)
        content = fetched_cache[cache_key]
        if not content:
            return f"(could not fetch {path_match})"
        snippet_query = ref.detail.replace(path_match, "").lstrip(": ")
        return _snippet_around(content, snippet_query or ref.detail)
    return "(unknown kind)"


def _print_explain(report: EvidenceReport, github: GitHubProfile, substring: str, from_cache: bool) -> None:
    repo_map = {r.full_name.lower(): r for r in github.repos}
    fetched_cache: dict[str, str | None] = {}
    matches = [r for r in report.results if substring.lower() in r.claim_text.lower()]
    if not matches:
        print(f"No claims matched substring: {substring!r}")
        return

    for r in matches:
        print("=" * 70)
        print(f"Claim:    {r.claim_text} ({r.claim_type})")
        print(f"Project:  {r.project_name} | Repo: {r.matched_repo or '-'}")
        print(f"Verdict:  {r.verdict.value} | Model: {r.model}")
        print(f"Reason:   {r.reason}")
        if r.ref_relevance:
            print(f"Rel:      {r.ref_relevance}")
        if r.verification_answer:
            print(f"Verify:   {r.verification_answer}")
        if r.checked:
            print(f"Checked:  {r.checked}")
        if r.signals:
            print(f"Signals:  {r.signals}")
        if r.suggested_repo:
            print(f"Suggest:  {r.suggested_repo}")
        if r.candidate_action:
            print(f"Action:   {r.candidate_action}")
        repo = repo_map.get((r.matched_repo or "").lower())
        if not r.evidence:
            print("Evidence: []")
        else:
            print("Evidence:")
            for ref in r.evidence:
                snippet = _resolve_ref_snippet(ref, repo, fetched_cache)
                print(f"  - [{ref.kind}] (independent={ref.independent}) {ref.detail}")
                print(f"    -> {snippet}")

    cache_note = " (cached)" if from_cache else ""
    print(f"\nllm_calls={report.llm_calls}, fetch_file_calls={report.fetch_file_calls}{cache_note}")
    print(format_cache_summary(CACHE_DIR))


def _print_audit_strong(report: EvidenceReport, github: GitHubProfile, from_cache: bool) -> None:
    repo_map = {r.full_name.lower(): r for r in github.repos}
    fetched_cache: dict[str, str | None] = {}

    strong_non_tech = [
        r for r in report.results
        if r.verdict == Verdict.STRONG and r.claim_type != "technology"
    ]
    print(f"=== STRONG Non-Technology Claims ({len(strong_non_tech)}) ===")
    for r in strong_non_tech:
        repo = repo_map.get((r.matched_repo or "").lower())
        print("-" * 70)
        print(f"Claim:  {r.claim_text} [{r.claim_type}] ({r.project_name} -> {r.matched_repo or '-'})")
        print(f"Verify: {r.verification_answer or '(none)'} | ref_relevance={r.ref_relevance or '-'}")
        if r.signals:
            print(f"Signals: {r.signals}")
        for ref in r.evidence:
            path_label = _ref_path_label(ref, repo)
            snippet = _resolve_ref_snippet(ref, repo, fetched_cache)
            print(f"  [{ref.kind}] path={path_label} (independent={ref.independent})")
            print(f"    detail:  {ref.detail}")
            print(f"    snippet: {snippet}")

    flagged = [
        r for r in report.results
        if r.verdict != Verdict.STRONG and (r.verification_answer or r.ref_relevance == "low")
    ]
    if flagged:
        print(f"\n=== Downgraded by Verification / Relevance Guard ({len(flagged)}) ===")
        for r in flagged:
            print(f"- {r.claim_text} [{r.claim_type}] -> {r.verdict.value} | rel={r.ref_relevance or '-'} | verify={r.verification_answer or '-'}")

    cache_note = " (cached)" if from_cache else ""
    print(f"\nllm_calls={report.llm_calls}, fetch_file_calls={report.fetch_file_calls}{cache_note}")
    print(format_cache_summary(CACHE_DIR))


def _print_table(report: EvidenceReport, from_cache: bool) -> None:
    headers = ("Claim", "Repo", "Verdict")
    rows = []
    for r in report.results:
        claim_col = r.claim_text if len(r.claim_text) <= 68 else r.claim_text[:65] + "..."
        repo_col = r.matched_repo or "-"
        rows.append((claim_col, repo_col, r.verdict.value))

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))

    print("\nProject Gate Breakdown (feature-only gate):")
    for p in report.projects:
        print(
            f"- {p.project_name} ({p.repo or '-'}) | "
            f"STRONG={p.counts.get('STRONG', 0)} (indep_features={p.strong_independent_features}, all_features={p.strong_feature_claims}) | "
            f"MOD={p.counts.get('MODERATE', 0)} (features={p.moderate_feature_claims}) | "
            f"INSUFF={p.counts.get('INSUFFICIENT_EVIDENCE', 0)} "
            f"UNSUP={p.counts.get('UNSUPPORTED', 0)} "
            f"CONTR={p.counts.get('CONTRADICTED', 0)} | "
            f"evidenced={p.evidenced} ({p.evidenced_reason})"
        )

    print(
        f"\nEvidenced Projects ({len(report.evidenced_projects)}/{settings.min_projects_with_evidence} required; "
        f"min_moderate_features={settings.min_moderate_features}):"
    )
    if not report.evidenced_projects:
        print("- None")
    else:
        for ep in report.evidenced_projects:
            print(f"- {ep.project_name} ({ep.repo or '-'}): {ep.reason}")
            for q_claim in ep.qualifying_claims:
                print(f"    claim: {q_claim}")
            for q_ref in ep.qualifying_refs:
                print(f"    ref:   [{q_ref.kind}] (independent={q_ref.independent}) {q_ref.detail}")

    print(f"\nevidence_gate_passed: {report.evidence_gate_passed}")
    cache_note = " (loaded from cache)" if from_cache else ""
    print(f"llm_calls={report.llm_calls}, fetch_file_calls={report.fetch_file_calls}{cache_note}")
    print(format_cache_summary(CACHE_DIR))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", nargs="?", default=None)
    parser.add_argument("username", nargs="?", default=None)
    parser.add_argument("--resume-json", default=None)
    parser.add_argument("--github-json", default=None)
    parser.add_argument("--confirm-email", action="append", default=[])
    parser.add_argument("--inject-claim", action="append", default=[])
    parser.add_argument("--explain", default=None)
    parser.add_argument("--audit-strong", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-llm-cache", action="store_true")
    args = parser.parse_args()

    reset_llm_cache_stats()
    set_llm_cache_enabled(not args.no_llm_cache)

    skip_llm_artifact_cache = args.no_cache or args.no_llm_cache
    resume = _load_or_parse_resume(args.pdf_path, args.resume_json, skip_llm_artifact_cache)
    resume = _inject_claims(resume, args.inject_claim)
    github = _load_or_analyze_github(
        args.username,
        args.github_json,
        resume,
        args.confirm_email,
        args.no_cache,
    )

    cache_file = _report_cache_path(resume, github)
    report: EvidenceReport | None = None
    from_cache = False
    if not skip_llm_artifact_cache and cache_file.exists():
        try:
            report = EvidenceReport.model_validate_json(cache_file.read_text(encoding="utf-8"))
            from_cache = True
        except (ValidationError, ValueError):
            report = None

    if report is None:
        report = check_resume(resume, github)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if args.audit_strong:
        _print_audit_strong(report, github, from_cache=from_cache)
    elif args.explain:
        _print_explain(report, github, args.explain, from_cache=from_cache)
    else:
        _print_table(report, from_cache=from_cache)
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
