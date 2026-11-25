import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError
from backend.app.schemas import ResumeProfile
from backend.app.services.llm_client import (
    format_cache_summary,
    reset_llm_cache_stats,
    set_llm_cache_enabled,
)
from backend.app.services.pdf_extractor import extract_text
from backend.app.services.resume_parser import parse_resume

DEFAULT_RESUME = Path.home() / "Documents" / "resume.pdf"
CACHE_DIR = ROOT / "samples" / ".cache"


def _cache_path(pdf_path: Path) -> Path:
    resolved = str(pdf_path.expanduser().resolve())
    path_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{path_hash}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", nargs="?", default=str(DEFAULT_RESUME))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-llm-cache", action="store_true")
    args = parser.parse_args()

    reset_llm_cache_stats()
    set_llm_cache_enabled(not args.no_llm_cache)

    pdf_path = Path(args.pdf_path)
    cache_file = _cache_path(pdf_path)

    if not args.no_cache and not args.no_llm_cache and cache_file.exists():
        try:
            cached = ResumeProfile.model_validate_json(cache_file.read_text(encoding="utf-8"))
            print(cached.model_dump_json(indent=2))
            print(f"\nRun summary: source=cache | {format_cache_summary(CACHE_DIR)}")
            return
        except (ValidationError, ValueError):
            pass

    text = extract_text(pdf_path)
    profile = parse_resume(text)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_json = profile.model_dump_json(indent=2)
    cache_file.write_text(output_json, encoding="utf-8")
    print(output_json)
    print(f"\nRun summary: source=fresh | {format_cache_summary(CACHE_DIR)}")


if __name__ == "__main__":
    main()
