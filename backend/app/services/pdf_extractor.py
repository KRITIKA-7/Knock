from pathlib import Path
import pdfplumber


def extract_text(pdf_path: str | Path) -> str:
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

    result = "\n\n".join(pages).strip()
    if not result:
        raise ValueError(f"No extractable text found in {path}")
    return result
