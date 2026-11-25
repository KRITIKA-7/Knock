import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.pdf_extractor import extract_text

DEFAULT_RESUME = Path.home() / "Documents" / "resume.pdf"


def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESUME
    text = extract_text(pdf_path)
    print(f"File: {pdf_path}")
    print(f"Chars: {len(text)} | Lines: {len(text.splitlines())}")
    print("-" * 50)
    print(text)


if __name__ == "__main__":
    main()
