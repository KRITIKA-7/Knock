import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.schemas import FitResult, GitHubProfile, ResumeProfile, StartupProfile
from backend.app.services.fit_scorer import score_fit
from backend.app.services.llm_client import (
    format_cache_summary,
    reset_llm_cache_stats,
    set_llm_cache_enabled,
)
from backend.scripts.test_questions import CACHE_DIR, _load_or_run_report

STARTUPS_DIR = ROOT / "data" / "startups"


def _load_startups() -> list[StartupProfile]:
    files = sorted(STARTUPS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No startup JSON files found in {STARTUPS_DIR}")
    return [StartupProfile.model_validate_json(p.read_text(encoding="utf-8")) for p in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*")
    parser.add_argument("--resume-json", default=None)
    parser.add_argument("--github-json", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-llm-cache", action="store_true")
    args = parser.parse_args()

    reset_llm_cache_stats()
    set_llm_cache_enabled(not args.no_llm_cache)

    resume_path = args.resume_json
    github_path = args.github_json

    if len(args.args) == 2:
        resume_path, github_path = args.args
    elif args.args:
        raise ValueError("Pass <resume_json> <github_json> or use --resume-json/--github-json")

    if not resume_path or not github_path:
        raise ValueError("Missing required resume JSON or github JSON")

    resume = ResumeProfile.model_validate_json(
        Path(resume_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    github = GitHubProfile.model_validate_json(
        Path(github_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    startups = _load_startups()
    report, from_cache = _load_or_run_report(resume, github, no_cache=args.no_cache)

    scored: list[tuple[StartupProfile, FitResult]] = [
        (sp, score_fit(report, sp)) for sp in startups
    ]
    scored.sort(key=lambda item: (-item[1].overall, -item[1].domain_fit, -item[1].skill_fit, item[0].slug))
    total_llm_calls = sum(res.llm_calls for _, res in scored)

    print(
        f"Report source: {'cache' if from_cache else 'fresh'} | Startups: {len(scored)} | Total llm_calls: {total_llm_calls}"
    )
    print(format_cache_summary(CACHE_DIR))
    print("=" * 102)
    print(
        f"{'Rank':<5} {'Startup':<14} {'Domain':<40} {'Overall':>7} {'Domain':>7} {'Skill':>7} {'Weak?':<6} {'Gaps':>5} {'Calls':>5}"
    )
    print("-" * 102)
    for idx, (sp, res) in enumerate(scored, start=1):
        domain_short = sp.domain if len(sp.domain) <= 38 else sp.domain[:35] + "..."
        weak_flag = "YES" if res.weak_match else "NO"
        print(
            f"{idx:<5} {sp.slug:<14} {domain_short:<40} {res.overall:>7.4f} {res.domain_fit:>7.4f} {res.skill_fit:>7.4f} {weak_flag:<6} {len(res.gaps):>5} {res.llm_calls:>5}"
        )
    print("=" * 102)

    for idx, (sp, res) in enumerate(scored, start=1):
        print(
            f"\n#{idx} {sp.name} ({sp.slug}) | overall={res.overall:.4f} | domain_fit={res.domain_fit:.4f} | skill_fit={res.skill_fit:.4f} | weak_match={res.weak_match} | llm_calls={res.llm_calls} | model={res.model} | prompt={res.prompt_version}"
        )
        if res.tech_overlap_sources:
            src_summary = ", ".join(f"{k} <- {v}" for k, v in res.tech_overlap_sources.items())
            print(f"Verified Tech Overlap Sources: {src_summary}")
        else:
            print("Verified Tech Overlap Sources: None")
        print(f"Matches ({len(res.matches)}):")
        for m in res.matches:
            target = m.startup_problem_id or "tech_stack"
            if m.tech_overlap:
                overlap = ", ".join(
                    f"{t} ({m.tech_overlap_sources.get(t, '?')})" for t in m.tech_overlap
                )
            else:
                overlap = "-"
            print(f"  - [{m.relation}] {m.project_name} -> {target} | tech_overlap=[{overlap}]")
            print(f"    {m.explanation}")
            for ref in m.evidence_refs:
                print(f"      ref: [{ref.kind}] (independent={ref.independent}) {ref.detail}")
        print(f"Gaps ({len(res.gaps)}):")
        if res.gaps:
            for g in res.gaps:
                print(f"  - {g}")
        else:
            print("  - None")


if __name__ == "__main__":
    main()
