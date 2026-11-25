import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError
from backend.app.schemas import GitHubProfile, ResumeProfile
from backend.app.services.github_analyzer import analyze_github

CACHE_DIR = ROOT / "samples" / ".cache"


def _collect_resume_links(resume_json_path: Path) -> list[str]:
    profile = ResumeProfile.model_validate_json(resume_json_path.read_text(encoding="utf-8"))
    links: list[str] = []
    for exp in profile.experience:
        links.extend(exp.links)
    for proj in profile.projects:
        links.extend(proj.links)
    return links


def _cache_path(username: str, links: list[str], confirmed_emails: list[str]) -> Path:
    norm_emails = sorted({e.strip().lower() for e in confirmed_emails if e.strip()})
    key_material = f"{username.lower()}|{'|'.join(sorted(links))}|{'|'.join(norm_emails)}"
    key_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"github_{username.lower()}_{key_hash}.json"


def _print_table(profile: GitHubProfile) -> None:
    headers = ("Repo", "User", "Unlinked", "Other", "Active Days", "Filtered Share", "Raw Share")
    rows = []
    for r in profile.repos:
        filtered = f"{r.largest_commit_share:.4f}" if r.largest_commit_share is not None else "-"
        raw = f"{r.raw_largest_commit_share:.4f}" if r.raw_largest_commit_share is not None else "-"
        user_str = str(r.user_commit_count)
        if r.confirmed_email_commit_count:
            user_str = f"{r.user_commit_count} ({r.confirmed_email_commit_count} email)"
        rows.append((
            r.name,
            user_str,
            str(r.unlinked_commit_count),
            str(r.other_author_count),
            str(r.active_days),
            filtered,
            raw,
        ))

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("username")
    parser.add_argument("--resume-json", default=None)
    parser.add_argument("--confirm-email", action="append", default=[])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    links = _collect_resume_links(Path(args.resume_json).expanduser().resolve()) if args.resume_json else []
    cache_file = _cache_path(args.username, links, args.confirm_email)

    profile: GitHubProfile | None = None
    if not args.no_cache and cache_file.exists():
        try:
            profile = GitHubProfile.model_validate_json(cache_file.read_text(encoding="utf-8"))
        except (ValidationError, ValueError):
            profile = None

    if profile is None:
        profile = analyze_github(args.username, links, confirmed_emails=args.confirm_email)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    print(profile.model_dump_json(indent=2))
    _print_table(profile)


if __name__ == "__main__":
    main()
