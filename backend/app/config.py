import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

EXTRA_BINARY_EXTS = {
    ".pt",
    ".pth",
    ".onnx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".tiff",
    ".svg",
    ".bin",
    ".h5",
    ".hdf5",
    ".tflite",
    ".pb",
    ".mlmodel",
    ".safetensors",
}

PATCH_EXTS = {".patch", ".diff"}
PATCH_DIRS = {"patch", "patches", "diff", "diffs"}

EXTRA_LOCKFILES = {
    "go.sum",
    "pipfile.lock",
    "gemfile.lock",
    "npm-shrinkwrap.json",
    "shrinkwrap.json",
    "bun.lockb",
    "bun.lock",
}

EXTRA_MANIFEST_FILES = {
    "cargo.toml",
    "go.mod",
    "gemfile",
    "pipfile",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "cmakelists.txt",
    "makefile",
    "setup.py",
    "setup.cfg",
    "app.json",
    "eas.json",
    "phpunit.xml",
    "artisan",
    ".editorconfig",
    ".gitignore",
    ".gitattributes",
    ".dockerignore",
    ".prettierrc",
    ".eslintrc",
}

DATA_EXTS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".jsonl",
    ".ndjson",
    ".arff",
    ".mat",
    ".feather",
    ".arrow",
    ".sql",
    ".tex",
}

DATA_AND_ASSET_DIRS = {
    "data",
    "dataset",
    "datasets",
    "testdata",
    "test_data",
    "samples",
    "assets",
    "weights",
    "checkpoints",
    "runs",
    "models",
    "figures",
    "storage",
}

BINARY_DETAIL_PATTERN = re.compile(
    r"\.(?:pt|pth|onnx|png|jpg|jpeg|gif|webp|ico|bmp|tiff|svg|bin|h5|hdf5|tflite|pb|mlmodel|safetensors|pkl|pickle|npy|npz|joblib|sqlite|db|exe|dll|so|dylib|pdf|zip|tar|gz|mp3|mp4|wav|flac)\b",
    re.IGNORECASE,
)

CONTROL_OR_DECL_PREFIX = re.compile(
    r"^(?:class|def|async\s+def|function|public\s+function|private\s+function|protected\s+function|static\s+function|if|elif|for|while|switch|return)\b"
)

SCALAR_KV_PATTERN = re.compile(
    r'^["\']?[\w.-]+["\']?\s*:\s*(?:-?\d+(?:\.\d+)?|true|false|null|none|["\'][^"\']*["\'])\s*,?$',
    re.IGNORECASE,
)


class Settings(BaseSettings):
    env: str = "development"
    gemini_api_key: str = ""
    gemini_model: str = ""
    verifier_model: str = ""
    scorer_model: str = ""
    github_token: str = ""
    database_url: str = f"sqlite:///{BASE_DIR}/knock.db"

    min_strong_claims: int = 1
    min_moderate_claims: int = 2
    min_moderate_features: int = 3
    min_projects_with_evidence: int = 2
    min_test_score: float = 0.70
    min_test_score_with_dispute: float = 0.85
    followup_weight: float = 0.3
    dimension_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "specificity_to_own_design": 0.35,
            "tradeoff_reasoning": 0.25,
            "technical_correctness": 0.25,
            "problem_understanding": 0.15,
        }
    )
    max_unlocks_per_week: int = 3
    max_unlocks_per_startup_per_week: int = 5
    max_resume_pdf_bytes: int = 5 * 1024 * 1024
    min_verified_questions: int = 2
    max_questions: int = 3
    llm_cache_enabled: bool = True

    fit_strong_weight: float = 1.0
    fit_moderate_weight: float = 0.6
    fit_relation_direct_weight: float = 1.0
    fit_relation_adjacent_weight: float = 0.6
    fit_relation_transferable_weight: float = 0.3
    fit_skill_overlap_weight: float = 0.5
    fit_skill_relation_weight: float = 0.5
    fit_domain_weight: float = 0.55
    fit_skill_weight: float = 0.45
    fit_language_overlap_weight: float = 0.1
    weak_match_relations: tuple[str, ...] = ("direct", "adjacent")

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def GEMINI_API_KEY(self) -> str:
        return self.gemini_api_key

    @property
    def GEMINI_MODEL(self) -> str:
        return self.gemini_model

    @property
    def VERIFIER_MODEL(self) -> str:
        return self.verifier_model or self.gemini_model

    @property
    def SCORER_MODEL(self) -> str:
        return self.scorer_model or self.gemini_model

    @property
    def GITHUB_TOKEN(self) -> str:
        return self.github_token

    @property
    def is_production(self) -> bool:
        raw = os.environ.get("ENV")
        val = raw if raw is not None else self.env
        return val.strip().lower() == "production"


def is_production_env() -> bool:
    return settings.is_production


settings = Settings()


@dataclass
class UsableGroundingRef:
    ref: Any
    file_path: str
    snippet: str


def split_fetched_detail(detail: str) -> tuple[str, str] | None:
    if ": " in detail:
        left, right = detail.split(": ", 1)
    elif ":" in detail:
        left, right = detail.split(":", 1)
    else:
        return None
    path = left.strip()
    snippet = right.strip()
    if not path or not snippet:
        return None
    return path, snippet


def is_one_line_constant_or_assignment(snippet: str) -> bool:
    lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
    if len(lines) != 1:
        return False
    line = lines[0]
    if line.startswith("#define "):
        return True
    if SCALAR_KV_PATTERN.match(line):
        return True
    if CONTROL_OR_DECL_PREFIX.match(line) or "=>" in line or line.endswith("{"):
        return False
    stripped_cmp = (
        line.replace("==", "")
        .replace("!=", "")
        .replace("<=", "")
        .replace(">=", "")
        .replace("=>", "")
    )
    return "=" in stripped_cmp


def is_disallowed_grounding_path(path: str, context: str = "") -> bool:
    from backend.app.services.evidence_checker import (
        LOCKFILES,
        SKIP_FETCH_EXTS,
        _is_weak_feature_path,
    )

    posix_path = PurePosixPath(path)
    basename = posix_path.name.lower()
    ext = posix_path.suffix.lower()
    dir_parts = {p.lower() for p in posix_path.parts[:-1]}

    if ext in (SKIP_FETCH_EXTS | EXTRA_BINARY_EXTS):
        return True
    if ext in PATCH_EXTS or (dir_parts & PATCH_DIRS):
        return True
    if basename in LOCKFILES or basename in EXTRA_LOCKFILES or basename.endswith(".lock"):
        return True
    if basename in EXTRA_MANIFEST_FILES:
        return True
    if _is_weak_feature_path(path, context):
        return True
    return False


def is_allowed_tree_file(path: str) -> bool:
    from backend.app.services.evidence_checker import WEAK_TEST_DIRS

    if is_disallowed_grounding_path(path, ""):
        return False
    posix_path = PurePosixPath(path)
    basename = posix_path.name.lower()
    ext = posix_path.suffix.lower()
    dir_parts = {p.lower() for p in posix_path.parts[:-1]}

    if ext in DATA_EXTS or (dir_parts & DATA_AND_ASSET_DIRS):
        return False
    if basename.startswith(".") or basename == ".gitkeep":
        return False
    if "manifest" in basename:
        return False
    is_test_file = (
        bool(dir_parts & WEAK_TEST_DIRS)
        or basename.startswith("test_")
        or basename.endswith(
            ("_test.py", ".test.js", ".test.ts", ".test.jsx", ".test.tsx", ".spec.js", ".spec.ts", ".spec.py", "testcase.php")
        )
        or basename.endswith("test.php")
    )
    if is_test_file:
        return False
    return True


def extract_usable_grounding_ref(ref: Any, context: str = "") -> UsableGroundingRef | None:
    if getattr(ref, "kind", None) != "fetched_file":
        return None
    detail = getattr(ref, "detail", "")
    if BINARY_DETAIL_PATTERN.search(detail):
        return None
    split = split_fetched_detail(detail)
    if split is None:
        return None
    path, snippet = split
    if is_disallowed_grounding_path(path, context):
        return None
    if is_one_line_constant_or_assignment(snippet):
        return None
    return UsableGroundingRef(ref=ref.model_copy(), file_path=path, snippet=snippet)


def has_usable_grounding(match: Any, context: str = "") -> bool:
    ctx = context or getattr(match, "explanation", "") or ""
    for ref in getattr(match, "evidence_refs", []) or []:
        if extract_usable_grounding_ref(ref, ctx) is not None:
            return True
    return False


def is_weak_match(
    matches: Iterable[Any],
    valid_problem_ids: set[str] | None = None,
) -> bool:
    for m in matches:
        if getattr(m, "relation", None) not in settings.weak_match_relations:
            continue
        prob_id = getattr(m, "startup_problem_id", None)
        if not prob_id:
            continue
        if valid_problem_ids is not None and prob_id not in valid_problem_ids:
            continue
        if has_usable_grounding(m):
            return False
    return True
