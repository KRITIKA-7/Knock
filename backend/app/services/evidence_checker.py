import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlparse

import httpx
from google.genai import errors
from pydantic import BaseModel, Field, ValidationError

from backend.app.config import settings
from backend.app.schemas import (
    Claim,
    ClaimResult,
    EvidenceRef,
    EvidenceReport,
    EvidencedProject,
    GitHubProfile,
    ProjectSummary,
    RepoEvidence,
    ResumeProfile,
    Verdict,
)
from backend.app.services.github_analyzer import fetch_file
from backend.app.services.llm_client import call_llm, invalidate_llm_cache

PROMPT_VERSION = "v5"
MAX_PASS1_FILES = 5

SYSTEM_PROMPT = (
    "You evaluate resume claims for a project against factual repository evidence and return ONLY valid JSON. "
    "Never invent evidence. Never use the phrase 'proof of authorship'."
)

PASS1_SYSTEM_PROMPT = (
    "You select the most informative file paths from a repository file tree to verify non-technology claims. "
    "Return ONLY valid JSON."
)

VERIFY_SYSTEM_PROMPT = (
    "You strictly verify whether a code or repository snippet shows a resume claim is implemented. "
    "Return ONLY valid JSON."
)

PASS1_SCHEMA_DESC = '{"files": [string]}'

PASS2_SCHEMA_DESC = """{
  "results": [
    {
      "claim_index": int,
      "verdict": "STRONG" | "MODERATE" | "INSUFFICIENT_EVIDENCE" | "UNSUPPORTED" | "CONTRADICTED",
      "evidence": [
        {
          "kind": "dependency" | "language" | "file" | "readme" | "commit_stat" | "homepage" | "fetched_file",
          "detail": string
        }
      ],
      "checked": [string],
      "reason": string,
      "signals": [string],
      "candidate_action": string | null
    }
  ]
}"""

VERIFY_SCHEMA_DESC = """{
  "verifications": [
    {
      "claim_index": int,
      "answer": "YES" | "PARTIAL" | "NO",
      "sentence": string
    }
  ]
}"""

PACKAGE_OR_README_FILES = {
    "readme",
    "readme.md",
    "readme.txt",
    "readme.rst",
    "requirements.txt",
    "package.json",
    "package-lock.json",
    "pyproject.toml",
    "composer.json",
    "composer.lock",
    "yarn.lock",
    "poetry.lock",
    "pnpm-lock.yaml",
}
LOCKFILES = {
    "package-lock.json",
    "yarn.lock",
    "composer.lock",
    "poetry.lock",
    "pnpm-lock.yaml",
    "cargo.lock",
}
SKIP_FETCH_EXTS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".pt",
    ".pth",
    ".onnx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp3",
    ".mp4",
    ".wav",
    ".flac",
    ".npy",
    ".npz",
    ".joblib",
    ".pkl",
    ".pickle",
    ".sqlite",
    ".db",
}
SKIP_FETCH_DIRS = {"node_modules", "vendor", "dist", "build", ".venv", "venv", ".git"}
DEPLOY_CONFIG_FILES = {"vercel.json", "render.yaml", "dockerfile", "netlify.toml", "procfile"}
ARTIFACT_DIRS = {"reports", "results", "logs", "outputs", "benchmarks", "notebooks"}
ARTIFACT_EXTS = {".ipynb", ".log"}
ARTIFACT_TEXT_EXTS = {".json", ".csv", ".md", ".txt"}
ARTIFACT_NAME_TOKENS = ("metric", "benchmark", "eval", "result", "report", "audit", "summary", "phase")

WEAK_LOCALE_DIRS = {"locale", "locales", "i18n", "l10n", "lang", "langs", "translations"}
WEAK_DOCS_DIRS = {"doc", "docs", "documentation", "wiki"}
WEAK_DOCS_EXTS = {".md", ".rst", ".adoc", ".txt", ".pdf", ".docx", ".pptx"}
WEAK_CONFIG_FILES = {
    "package.json",
    "package-lock.json",
    "composer.json",
    "composer.lock",
    "requirements.txt",
    "pyproject.toml",
    "poetry.lock",
    "yarn.lock",
    "pnpm-lock.yaml",
    "cargo.lock",
    "manifest.json",
    "tsconfig.json",
    "jsconfig.json",
    "vite.config.js",
    "vite.config.ts",
    "tailwind.config.js",
    "tailwind.config.ts",
    "postcss.config.js",
    "next.config.js",
    "next.config.mjs",
    "webpack.config.js",
    "babel.config.js",
    "eslint.config.js",
    ".eslintrc.json",
    "docker-compose.yml",
    "docker-compose.yaml",
    "dockerfile",
    "vercel.json",
    "render.yaml",
    "netlify.toml",
    "procfile",
    ".env.example",
    "artisan",
}
WEAK_CONFIG_DIRS = {"config", "configs", "settings"}
WEAK_CONFIG_EXTS = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env"}
WEAK_SNAPSHOT_FIXTURE_DIRS = {
    "snapshot",
    "snapshots",
    "__snapshots__",
    "fixture",
    "fixtures",
    "__fixtures__",
    "mock",
    "mocks",
    "__mocks__",
    "testdata",
    "test_data",
}
WEAK_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs"}

AUTHORSHIP_PATTERNS = [
    re.compile(
        r"\b(?:independently|solely|single[- ]handedly|exclusively|personally|individually|spearheaded|pioneered|championed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\b)(?:led)\s+(?:a|an|the|development|design|team|project|effort|migration|build|implementation)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^led\b", re.IGNORECASE),
    re.compile(r"\b(?:on my own|by myself)\b", re.IGNORECASE),
]

STOPWORDS = {
    "built",
    "build",
    "used",
    "using",
    "use",
    "implemented",
    "implement",
    "integrated",
    "integrate",
    "developed",
    "develop",
    "engineered",
    "engineer",
    "shipped",
    "ship",
    "created",
    "create",
    "designed",
    "design",
    "added",
    "add",
    "achieved",
    "achieve",
    "mitigated",
    "mitigate",
    "eliminated",
    "eliminate",
    "conducted",
    "conduct",
    "uncovered",
    "uncover",
    "processed",
    "process",
    "evaluated",
    "evaluate",
    "deployed",
    "deploy",
    "cut",
    "resulting",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "via",
    "across",
    "over",
    "under",
    "based",
    "compared",
    "system",
    "platform",
    "application",
    "app",
    "project",
    "portal",
    "engine",
    "client",
    "backend",
    "frontend",
    "feature",
    "features",
    "support",
    "including",
}


class _Pass1Selection(BaseModel):
    files: list[str] = Field(default_factory=list)


class _LLMClaimVerdict(BaseModel):
    claim_index: int
    verdict: Verdict
    evidence: list[EvidenceRef] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)
    reason: str
    signals: list[str] = Field(default_factory=list)
    candidate_action: str | None = None


class _LLMBatchOutput(BaseModel):
    results: list[_LLMClaimVerdict] = Field(default_factory=list)


class _VerifyItem(BaseModel):
    claim_index: int
    answer: Literal["YES", "PARTIAL", "NO"]
    sentence: str


class _VerifyBatchOutput(BaseModel):
    verifications: list[_VerifyItem] = Field(default_factory=list)


def _strip_authorship_words(text: str) -> tuple[str, bool]:
    cleaned = text
    found = False
    for pattern in AUTHORSHIP_PATTERNS:
        if pattern.search(cleaned):
            found = True
            cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;-")
    if not cleaned:
        cleaned = text
    return cleaned, found


def _parse_github_repo_key(link: str) -> str | None:
    raw = link.strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None
    return f"{owner}/{repo}".lower()


def _match_linked_repo(links: list[str], repo_map: dict[str, RepoEvidence]) -> RepoEvidence | None:
    first_matched: RepoEvidence | None = None
    for link in links:
        key = _parse_github_repo_key(link)
        if not key:
            continue
        repo = repo_map.get(key)
        if repo is None:
            continue
        if first_matched is None:
            first_matched = repo
        if repo.fetch_error is None:
            return repo
    return first_matched


def _token_in_text(token: str, text: str) -> bool:
    t = token.strip().lower()
    if not t:
        return False
    return bool(re.search(rf"(?<![a-z0-9_.\-]){re.escape(t)}(?![a-z0-9_.\-])", text.lower()))


def _find_mismatched_repo(
    claim: Claim,
    project_name: str,
    matched_repo: RepoEvidence,
    all_repos: list[RepoEvidence],
) -> RepoEvidence | None:
    matched_key = matched_repo.full_name.lower()
    matched_name_low = matched_repo.name.lower()
    for repo in all_repos:
        if repo.full_name.lower() == matched_key:
            continue
        if repo.name.lower() == matched_name_low:
            continue
        for needle in (repo.full_name, repo.name):
            if len(needle) < 3:
                continue
            if _token_in_text(needle, claim.source_excerpt) or _token_in_text(needle, project_name):
                return repo
    return None


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _elapsed_hours(first_ts: str | None, last_ts: str | None) -> float | None:
    if not first_ts or not last_ts:
        return None
    try:
        dt1 = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        dt2 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        return round(abs((dt2 - dt1).total_seconds()) / 3600.0, 2)
    except ValueError:
        return None


def _build_repo_payload(repo: RepoEvidence) -> dict:
    data = repo.model_dump()
    data["unlinked_identities"] = [
        {"name": ident.name}
        for ident in repo.unlinked_identities
    ]
    data["elapsed_hours"] = _elapsed_hours(repo.first_commit_at, repo.last_commit_at)
    return data


def _is_fetchable_candidate(path: str) -> bool:
    posix_path = PurePosixPath(path)
    basename = posix_path.name.lower()
    if basename in LOCKFILES:
        return False
    dir_parts = {p.lower() for p in posix_path.parts[:-1]}
    if dir_parts & SKIP_FETCH_DIRS:
        return False
    if posix_path.suffix.lower() in SKIP_FETCH_EXTS:
        return False
    if any(basename.endswith(s) for s in (".min.js", ".min.css", ".map")):
        return False
    return True


def _is_test_claim(claim_text: str) -> bool:
    return bool(
        re.search(
            r"\b(test|tests|tested|testing|vitest|pytest|jest|phpunit|unittest|unit test|integration test)\b",
            claim_text.lower(),
        )
    )


def _is_weak_feature_path(path: str, claim_text: str) -> bool:
    posix_path = PurePosixPath(path)
    basename = posix_path.name.lower()
    ext = posix_path.suffix.lower()
    dir_parts = {p.lower() for p in posix_path.parts[:-1]}

    if dir_parts & WEAK_LOCALE_DIRS:
        return True
    if any(tok in basename for tok in ("translation", "locale", "i18n")):
        return True

    if dir_parts & WEAK_DOCS_DIRS or ext in WEAK_DOCS_EXTS or basename.startswith("readme"):
        return True

    if any(tok in basename for tok in ("changelog", "release_notes", "history")):
        return True

    if (
        basename in WEAK_CONFIG_FILES
        or (dir_parts & WEAK_CONFIG_DIRS)
        or ext in WEAK_CONFIG_EXTS
        or ".config." in basename
        or basename.startswith("config.")
    ):
        return True

    if (
        (dir_parts & WEAK_SNAPSHOT_FIXTURE_DIRS)
        or basename.endswith(".snap")
        or "snapshot" in basename
        or "fixture" in basename
    ):
        return True

    is_test_file = (
        bool(dir_parts & WEAK_TEST_DIRS)
        or basename.startswith("test_")
        or basename.endswith(("_test.py", ".test.js", ".test.ts", ".test.jsx", ".test.tsx", ".spec.js", ".spec.ts", ".spec.py"))
    )
    if is_test_file and not _is_test_claim(claim_text):
        return True

    return False


def _extract_ref_path(ref: EvidenceRef, repo: RepoEvidence) -> str | None:
    if ref.kind == "fetched_file":
        for p in repo.file_tree:
            if p in ref.detail:
                return p
        return ref.detail.split(":", 1)[0].strip() or None
    if ref.kind == "file":
        for p in repo.file_tree:
            if p in ref.detail:
                return p
        return ref.detail.strip() or None
    return None


def _build_pass1_prompt(claims: list[tuple[int, Claim, str]], file_tree: list[str]) -> str:
    non_tech = [
        {"text": stripped, "type": c.type}
        for _, c, stripped in claims
        if c.type != "technology"
    ]
    return f"""Select up to {MAX_PASS1_FILES} exact file paths from file_tree that are most useful for verifying the non-technology claims below (feature implementations, metrics/results/evaluation logs, deployment configs).

Return ONLY JSON matching:
{PASS1_SCHEMA_DESC}

Rules:
- Return at most {MAX_PASS1_FILES} exact paths from file_tree.
- For feature claims, choose actual implementation source code files (controllers, services, modules, components)—NOT locale/translation/i18n files, docs/markdown files, changelogs, snapshots, or fixtures. Choose test files only if the claim is about testing.
- For numeric_metric claims, choose metrics/benchmark JSON, evaluation logs, or audit reports if present.
- For deployment claims, include deploy config files (Dockerfile, render.yaml, vercel.json, etc.) if present.
- Never invent paths not in file_tree.

Input JSON:
{json.dumps({"non_technology_claims": non_tech, "file_tree": file_tree}, ensure_ascii=False)}"""


def _call_checker_llm(
    prompt: str | list[dict[str, str]],
    system: str,
    models_used: set[str],
    counters: dict[str, int],
) -> tuple[str, str]:
    for attempt in range(2):
        used_model = settings.gemini_model
        models_used.add(used_model)
        counters["llm_calls"] += 1
        try:
            return call_llm(prompt, system=system, temperature=0), used_model
        except errors.APIError as err:
            msg = str(err)
            if err.code in (429, 503) and attempt == 0 and "PerDay" not in msg:
                delay = 20
                m = re.search(r"retry\s+in\s+(\d+(?:\.\d+)?)s", msg, flags=re.IGNORECASE)
                if m:
                    delay = min(max(int(float(m.group(1))) + 2, 5), 40)
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("checker llm unreachable")


def _select_and_fetch_files(
    claims: list[tuple[int, Claim, str]],
    repo: RepoEvidence,
    models_used: set[str],
    counters: dict[str, int],
) -> dict[str, str]:
    if not repo.file_tree or not any(c.type != "technology" for _, c, _ in claims):
        return {}

    prompt = _build_pass1_prompt(claims, repo.file_tree)
    raw, _ = _call_checker_llm(prompt, PASS1_SYSTEM_PROMPT, models_used, counters)

    try:
        selection = _Pass1Selection.model_validate_json(_strip_code_fences(raw))
    except (ValidationError, ValueError) as first_err:
        invalidate_llm_cache(prompt, system=PASS1_SYSTEM_PROMPT, temperature=0)
        retry_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": f"Validation failed: {first_err}. Return ONLY JSON matching {PASS1_SCHEMA_DESC}.",
            },
        ]
        retry_raw, _ = _call_checker_llm(retry_messages, PASS1_SYSTEM_PROMPT, models_used, counters)
        try:
            selection = _Pass1Selection.model_validate_json(_strip_code_fences(retry_raw))
        except (ValidationError, ValueError):
            invalidate_llm_cache(retry_messages, system=PASS1_SYSTEM_PROMPT, temperature=0)
            raise

    allowed = set(repo.file_tree)
    seen: set[str] = set()
    chosen: list[str] = []
    for path in selection.files:
        if path in allowed and path not in seen and _is_fetchable_candidate(path):
            seen.add(path)
            chosen.append(path)
            if len(chosen) >= MAX_PASS1_FILES:
                break

    fetched: dict[str, str] = {}
    for path in chosen:
        counters["fetch_file_calls"] += 1
        content = fetch_file(repo.full_name, path, max_bytes=20000)
        if content is not None:
            fetched[path] = content
    return fetched


def _build_pass2_prompt(
    claims: list[tuple[int, Claim, str]],
    repo_payload: dict,
    fetched_files: dict[str, str],
) -> str:
    input_obj: dict = {
        "claims": [
            {
                "claim_index": idx,
                "text": stripped,
                "type": claim.type,
                "source_excerpt": claim.source_excerpt,
            }
            for idx, claim, stripped in claims
        ],
        "repo": repo_payload,
        "fetched_files": fetched_files,
    }

    return f"""Evaluate every resume claim below for this single repository using ONLY the provided repository evidence and fetched_files.

Return ONLY JSON matching this schema:
{PASS2_SCHEMA_DESC}

Rules:
- Return one entry in "results" per claim_index in "claims" (exact matching claim_index).
- Every result MUST include a non-empty "checked" list naming the exact repository sources examined (e.g. ["dependencies", "file_tree", "readme_excerpt", "fetched_files", "commit_stats"]).
- Cite ONLY evidence refs that directly verify THAT SPECIFIC claim (never cite an unrelated snippet about a different feature or metric).
- Technology claims: check dependencies, languages, file_tree, and readme_excerpt. Dependency present plus matching files = STRONG. Dependency or language only = MODERATE.
- All non-technology claims ("feature", "numeric_metric", "deployment", "timeline", "other"): STRONG requires concrete independent evidence (such as fetched_files source/metrics content, repo results/metrics files, live homepage + deploy config, or commit_stat). README-only evidence caps at MODERATE.
- Numeric metrics (claim type "numeric_metric": accuracy, N=, latency, percentages, dataset sizes): verdict MUST be INSUFFICIENT_EVIDENCE unless readme_excerpt or fetched_files explicitly shows the same verbatim number.
- Timeline claims (claim type "timeline"): judge ONLY by elapsed_hours between first_commit_at and last_commit_at (cite kind "commit_stat"). If elapsed_hours fits the claimed duration (e.g. <= 36 hours for "in 1 day"), verdict is STRONG; if elapsed_hours clearly exceeds the claimed timeframe, verdict is CONTRADICTED; if elapsed_hours is null, verdict is INSUFFICIENT_EVIDENCE. Do NOT require the duration in readme_excerpt and do NOT use active_days.
- Deployment claims: STRONG only if repo.homepage is set AND a deploy config file exists in file_tree (vercel.json, render.yaml, Dockerfile, netlify.toml, Procfile). Otherwise MODERATE or INSUFFICIENT_EVIDENCE.
- Feature claims: STRONG only when backed by fetched_files implementation source code (cite kind "fetched_file" on a real implementation file—NOT locale/i18n, docs, changelog, config, snapshot, fixture, or unrelated test files). Without implementation code in fetched_files, verdict MUST be MODERATE.
- UNSUPPORTED is allowed ONLY when file_tree_truncated is false, readme_excerpt is non-empty, the claim type is "technology" or "feature", and nothing backs the claim. Otherwise use INSUFFICIENT_EVIDENCE.
- CONTRADICTED only for affirmative conflicting evidence in the repo (e.g. claims React, while dependencies and readme_excerpt show Vue only).
- If user_commit_count == 0 and unlinked_commit_count > 0: include "unlinked commits, candidate must confirm identity" in signals; NEVER use UNSUPPORTED or CONTRADICTED for unlinked commits.
- Put is_fork, largest_commit_share, raw_largest_commit_share, active_days, and other_author_count observations in signals as plain factual notes. They may lower confidence (e.g., MODERATE instead of STRONG) or inform candidate_action, but never alone cause UNSUPPORTED or CONTRADICTED.
- Every non-INSUFFICIENT_EVIDENCE verdict (including UNSUPPORTED and CONTRADICTED) MUST cite at least one EvidenceRef taken verbatim from repo.* or fetched_files.* (NEVER copy text from claim.source_excerpt into EvidenceRef.detail; for UNSUPPORTED, cite the examined manifest/README/file such as package.json or composer.json where the claimed item is absent):
  - kind "dependency": exact package name from repo.dependencies
  - kind "language": exact language key from repo.languages
  - kind "file": exact path from repo.file_tree
  - kind "readme": exact verbatim substring appearing inside repo.readme_excerpt (omit kind "readme" if not verbatim inside repo.readme_excerpt)
  - kind "homepage": exact repo.homepage URL
  - kind "commit_stat": exact commit field values from repo (include exact timestamps or numbers)
  - kind "fetched_file": "<exact_path>: <exact verbatim substring from fetched_files>"
- Never invent evidence. Never write "proof of authorship".

Input JSON:
{json.dumps(input_obj, ensure_ascii=False)}"""


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _text_matches_source(detail: str, source_text: str) -> bool:
    norm_source = _normalize_ws(source_text)
    cleaned = re.sub(r"^(readme\s*:?\s*)", "", detail, flags=re.IGNORECASE).strip(" \"'`")
    norm_detail = _normalize_ws(cleaned)
    if not norm_detail:
        return False
    if norm_detail in norm_source:
        return True
    chunks = [c.strip() for c in norm_detail.split("...") if len(c.strip()) >= 6]
    if chunks and all(c in norm_source for c in chunks):
        return True

    numbers = re.findall(r"\d+(?:[.,]\d+)?", norm_detail)
    if numbers and not all(num in norm_source for num in numbers):
        return False

    words = [
        w for w in re.findall(r"[a-z0-9_.\-]{4,}", norm_detail)
        if w not in {"readme", "mentions", "shows", "lists", "describes", "section", "file", "code", "implements", "uses", "with", "from", "that", "this"}
    ]
    if not words:
        return bool(numbers)
    hits = sum(1 for w in words if w in norm_source)
    return hits / len(words) >= 0.6


def _verify_single_ref(
    ref: EvidenceRef,
    repo: RepoEvidence,
    fetched_files: dict[str, str],
) -> bool:
    detail = ref.detail.strip()
    if not detail:
        return False

    if ref.kind == "dependency":
        return any(_token_in_text(dep, detail) for dep in repo.dependencies)

    if ref.kind == "language":
        return any(_token_in_text(lang, detail) for lang in repo.languages.keys())

    if ref.kind == "file":
        return any(path in detail for path in repo.file_tree)

    if ref.kind == "homepage":
        if not repo.homepage:
            return False
        hp = repo.homepage.strip().lower().rstrip("/")
        d_low = detail.lower().rstrip("/")
        return hp in d_low or urlparse(hp).netloc in d_low

    if ref.kind == "readme":
        if not repo.readme_excerpt:
            return False
        return _text_matches_source(detail, repo.readme_excerpt)

    if ref.kind == "fetched_file":
        if not fetched_files:
            return False
        for path, content in fetched_files.items():
            if path in detail:
                remainder = detail.replace(path, "").lstrip(": ").strip(" \"'`")
                if not remainder:
                    return True
                return _text_matches_source(remainder, content)
        return False

    if ref.kind == "commit_stat":
        if repo.total_commit_count == 0:
            return False
        stat_values: list[str] = [
            str(repo.user_commit_count),
            str(repo.unlinked_commit_count),
            str(repo.other_author_count),
            str(repo.confirmed_email_commit_count),
            str(repo.total_commit_count),
            str(repo.active_days),
        ]
        if repo.first_commit_at:
            stat_values.append(repo.first_commit_at.lower())
            stat_values.append(repo.first_commit_at[:10].lower())
        if repo.last_commit_at:
            stat_values.append(repo.last_commit_at.lower())
            stat_values.append(repo.last_commit_at[:10].lower())
        if repo.largest_commit_share is not None:
            stat_values.append(str(repo.largest_commit_share))
        if repo.raw_largest_commit_share is not None:
            stat_values.append(str(repo.raw_largest_commit_share))
        if repo.commits_ahead_of_parent is not None:
            stat_values.append(str(repo.commits_ahead_of_parent))
        if repo.merged_prs_to_parent is not None:
            stat_values.append(str(repo.merged_prs_to_parent))
        elapsed = _elapsed_hours(repo.first_commit_at, repo.last_commit_at)
        if elapsed is not None:
            stat_values.append(str(elapsed))
            stat_values.append(str(round(elapsed)))
            stat_values.append(f"{elapsed:.1f}")

        d_low = detail.lower()
        return any(_token_in_text(val, d_low) or val in d_low for val in stat_values if val)

    return False


def _is_artifact_file_path(path: str) -> bool:
    posix_path = PurePosixPath(path)
    basename = posix_path.name.lower()
    if basename in PACKAGE_OR_README_FILES or basename.startswith("readme"):
        return False
    dir_parts = {p.lower() for p in posix_path.parts[:-1]}
    if dir_parts & ARTIFACT_DIRS:
        return True
    ext = posix_path.suffix.lower()
    if ext in ARTIFACT_EXTS:
        return True
    if ext in ARTIFACT_TEXT_EXTS and any(tok in basename for tok in ARTIFACT_NAME_TOKENS):
        return True
    return False


def _has_deploy_config(file_tree: list[str]) -> bool:
    return any(PurePosixPath(p).name.lower() in DEPLOY_CONFIG_FILES for p in file_tree)


def _check_url_responds(url: str, url_cache: dict[str, bool]) -> bool:
    raw = url.strip()
    if not raw:
        return False
    if "://" not in raw:
        raw = "https://" + raw
    if raw in url_cache:
        return url_cache[raw]
    ok = False
    try:
        with httpx.Client(timeout=5.0, follow_redirects=True) as client:
            resp = client.head(raw)
            if resp.status_code == 405:
                resp = client.get(raw)
            ok = resp.status_code < 400
    except httpx.HTTPError:
        ok = False
    url_cache[raw] = ok
    return ok


def _compute_ref_independence(
    ref: EvidenceRef,
    repo: RepoEvidence,
    url_cache: dict[str, bool],
) -> bool:
    if ref.kind in ("readme", "dependency", "language"):
        return False
    if ref.kind in ("fetched_file", "commit_stat"):
        return True
    if ref.kind == "homepage":
        return bool(repo.homepage and _check_url_responds(repo.homepage, url_cache))
    if ref.kind == "file":
        matched_paths = [p for p in repo.file_tree if p in ref.detail]
        return any(_is_artifact_file_path(p) for p in matched_paths)
    return False


def _extract_numeric_tokens(text: str) -> list[str]:
    return re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", text)


def _cited_texts_for_numeric(
    evidence: list[EvidenceRef],
    repo: RepoEvidence,
    fetched_files: dict[str, str],
) -> list[str]:
    texts: list[str] = []
    for ref in evidence:
        if ref.kind == "readme" and repo.readme_excerpt and _verify_single_ref(ref, repo, fetched_files):
            texts.append(repo.readme_excerpt)
        elif ref.kind == "fetched_file" and _verify_single_ref(ref, repo, fetched_files):
            for path, content in fetched_files.items():
                if path in ref.detail:
                    texts.append(content)
    return texts


def _extract_key_terms(claim_text: str, project_name: str, repo_name: str) -> list[str]:
    ignore = set(STOPWORDS)
    for src in (project_name, repo_name):
        for w in re.findall(r"[a-z0-9]+", src.lower()):
            if len(w) >= 2:
                ignore.add(w)

    terms: list[str] = []
    seen: set[str] = set()
    for num in re.findall(r"\d+(?:\.\d+)?", claim_text.replace(",", "")):
        if num not in seen:
            seen.add(num)
            terms.append(num)
    for raw_word in re.findall(r"[a-zA-Z0-9]+", claim_text.lower()):
        if raw_word in ignore or len(raw_word) < 3:
            continue
        if raw_word not in seen:
            seen.add(raw_word)
            terms.append(raw_word)
    return terms


def _extract_snippet_window(source_text: str, query: str, radius: int = 220) -> str:
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


def _get_ref_snippet_for_check(
    ref: EvidenceRef,
    repo: RepoEvidence,
    fetched_files: dict[str, str],
) -> tuple[str, str]:
    if ref.kind == "fetched_file":
        path = _extract_ref_path(ref, repo) or "fetched_file"
        content = fetched_files.get(path, "")
        query = ref.detail.replace(path, "").lstrip(": ").strip()
        snippet = _extract_snippet_window(content, query or ref.detail) if content else ref.detail
        return path, f"{ref.detail}\n{snippet}".strip()
    if ref.kind == "readme":
        snippet = _extract_snippet_window(repo.readme_excerpt or "", ref.detail) if repo.readme_excerpt else ref.detail
        return "README.md", f"{ref.detail}\n{snippet}".strip()
    if ref.kind == "file":
        path = _extract_ref_path(ref, repo) or ref.detail
        return path, ref.detail
    if ref.kind == "homepage":
        return repo.homepage or "homepage", ref.detail
    if ref.kind == "commit_stat":
        return "git_commits", ref.detail
    return ref.kind, ref.detail


def _snippet_has_key_term(snippet: str, key_terms: list[str]) -> bool:
    if not key_terms:
        return True
    s_low = snippet.lower().replace(",", "")
    s_norm = re.sub(r"[^a-z0-9]+", " ", s_low)
    for term in key_terms:
        if term in s_low or term in s_norm:
            return True
        if len(term) >= 5 and term[:-1] in s_norm:
            return True
    return False


def _call_pass2_and_validate(
    prompt: str,
    expected_indices: set[int],
    models_used: set[str],
    counters: dict[str, int],
) -> tuple[_LLMBatchOutput, str]:
    raw, used_model = _call_checker_llm(prompt, SYSTEM_PROMPT, models_used, counters)

    def _validate(raw_str: str) -> _LLMBatchOutput:
        parsed = _LLMBatchOutput.model_validate_json(_strip_code_fences(raw_str))
        got_indices = {r.claim_index for r in parsed.results}
        missing = expected_indices - got_indices
        if missing:
            raise ValueError(f"Missing claim_index entries: {sorted(missing)}")
        for r in parsed.results:
            if not r.checked:
                raise ValueError(f"Empty checked list for claim_index {r.claim_index}")
        return parsed

    try:
        return _validate(raw), used_model
    except (ValidationError, ValueError) as first_err:
        invalidate_llm_cache(prompt, system=SYSTEM_PROMPT, temperature=0)
        retry_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"Validation failed with error:\n{first_err}\n\n"
                    "Return ONLY valid JSON matching the exact schema with a complete results list and non-empty checked lists."
                ),
            },
        ]
        retry_raw, used_model = _call_checker_llm(retry_messages, SYSTEM_PROMPT, models_used, counters)
        try:
            return _validate(retry_raw), used_model
        except (ValidationError, ValueError):
            invalidate_llm_cache(retry_messages, system=SYSTEM_PROMPT, temperature=0)
            raise


def _build_verify_prompt(items: list[dict]) -> str:
    return f"""For each claim and its cited snippet(s) below, answer strictly:
"Does this snippet show the claim is implemented? Answer YES, PARTIAL or NO with one sentence."

Return ONLY JSON matching:
{VERIFY_SCHEMA_DESC}

Rules:
- Return one verification object per claim_index in items.
- Answer YES only if the snippet concretely shows the claimed behavior/metric/deployment is implemented.
- Answer PARTIAL if it shows partial or indirect related logic without confirming the full claim.
- Answer NO if the snippet is unrelated, merely a label/doc/header, or does not implement the claim.

Input JSON:
{json.dumps({"items": items}, ensure_ascii=False)}"""


def _verify_strong_non_tech_claims(
    strong_results: list[tuple[int, ClaimResult, str]],
    repo: RepoEvidence,
    fetched_files: dict[str, str],
    models_used: set[str],
    counters: dict[str, int],
) -> None:
    if not strong_results:
        return

    verify_items: list[dict] = []
    for idx, res, stripped_text in strong_results:
        cited = []
        for ref in res.evidence:
            path, snippet = _get_ref_snippet_for_check(ref, repo, fetched_files)
            cited.append({"file_path": path, "snippet": snippet[:1200]})
        verify_items.append(
            {
                "claim_index": idx,
                "claim_text": stripped_text,
                "claim_type": res.claim_type,
                "cited_snippets": cited,
            }
        )

    prompt = _build_verify_prompt(verify_items)
    expected_indices = {idx for idx, _, _ in strong_results}

    def _validate_verify(raw_str: str) -> _VerifyBatchOutput:
        parsed = _VerifyBatchOutput.model_validate_json(_strip_code_fences(raw_str))
        got = {v.claim_index for v in parsed.verifications}
        missing = expected_indices - got
        if missing:
            raise ValueError(f"Missing verification indices: {sorted(missing)}")
        return parsed

    raw, _ = _call_checker_llm(prompt, VERIFY_SYSTEM_PROMPT, models_used, counters)

    try:
        parsed = _validate_verify(raw)
    except (ValidationError, ValueError) as first_err:
        invalidate_llm_cache(prompt, system=VERIFY_SYSTEM_PROMPT, temperature=0)
        retry_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": f"Validation failed: {first_err}. Return ONLY JSON matching {VERIFY_SCHEMA_DESC}.",
            },
        ]
        retry_raw, _ = _call_checker_llm(retry_messages, VERIFY_SYSTEM_PROMPT, models_used, counters)
        try:
            parsed = _validate_verify(retry_raw)
        except (ValidationError, ValueError):
            invalidate_llm_cache(retry_messages, system=VERIFY_SYSTEM_PROMPT, temperature=0)
            raise

    by_idx = {v.claim_index: v for v in parsed.verifications}
    for _, res, _ in strong_results:
        pass
    for idx, res, _ in strong_results:
        v = by_idx[idx]
        res.verification_answer = f"{v.answer}: {v.sentence.strip()}"
        if v.answer == "NO":
            res.verdict = Verdict.INSUFFICIENT_EVIDENCE
            res.reason = f"{res.reason} (downgraded: verification answered NO - {v.sentence.strip()})".strip()
        elif v.answer == "PARTIAL":
            res.verdict = Verdict.MODERATE
            res.reason = f"{res.reason} (capped: verification answered PARTIAL - {v.sentence.strip()})".strip()


def _sanitize_text(text: str) -> str:
    return re.sub(r"proof of authorship", "authorship evidence", text, flags=re.IGNORECASE)


def _finalize_claim_result(
    claim: Claim,
    stripped_text: str,
    had_authorship: bool,
    source_section: str,
    project_name: str,
    repo: RepoEvidence,
    llm_item: _LLMClaimVerdict,
    fetched_files: dict[str, str],
    model_name: str,
    url_cache: dict[str, bool],
) -> ClaimResult:
    verdict = llm_item.verdict
    evidence = [
        EvidenceRef(
            kind=ref.kind,
            detail=ref.detail,
            independent=_compute_ref_independence(ref, repo, url_cache),
        )
        for ref in llm_item.evidence
    ]
    checked = llm_item.checked
    reason = _sanitize_text(llm_item.reason.strip())
    signals = [_sanitize_text(s) for s in llm_item.signals]
    candidate_action = _sanitize_text(llm_item.candidate_action) if llm_item.candidate_action else None
    ref_relevance: str | None = None

    if had_authorship:
        auth_sig = "authorship claim cannot be verified from repo files"
        if not any(auth_sig in s.lower() for s in signals):
            signals.append(auth_sig)

    if verdict == Verdict.UNSUPPORTED:
        readme_ok = bool(repo.readme_excerpt and repo.readme_excerpt.strip())
        unsupported_allowed = (
            repo.fetch_error is None
            and not repo.file_tree_truncated
            and readme_ok
            and claim.type in ("technology", "feature")
        )
        if not unsupported_allowed:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            reason = f"{reason} (downgraded: UNSUPPORTED preconditions not met)".strip()

    if repo.user_commit_count == 0 and repo.unlinked_commit_count > 0:
        unlinked_msg = "unlinked commits, candidate must confirm identity"
        if not any(unlinked_msg in s.lower() for s in signals):
            signals.append(unlinked_msg)
        if verdict in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED):
            verdict = Verdict.INSUFFICIENT_EVIDENCE
        if not candidate_action:
            candidate_action = "confirm commit email identity"

    if verdict != Verdict.INSUFFICIENT_EVIDENCE:
        if not evidence or not all(_verify_single_ref(ref, repo, fetched_files) for ref in evidence):
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            reason = f"{reason} (downgraded: cited evidence did not match repo input)".strip()
            if not candidate_action:
                candidate_action = "link a public repo or explain in a dispute"

    if claim.type == "numeric_metric" and verdict != Verdict.INSUFFICIENT_EVIDENCE:
        claim_numbers = _extract_numeric_tokens(stripped_text)
        cited_texts = _cited_texts_for_numeric(evidence, repo, fetched_files)
        joined_cited = "\n".join(cited_texts)
        if not cited_texts or (claim_numbers and not all(num in joined_cited for num in claim_numbers)):
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            reason = f"{reason} (downgraded: claim numbers not found verbatim in cited README or fetched file)".strip()

    if claim.type == "deployment" and verdict == Verdict.STRONG:
        homepage_ok = bool(repo.homepage and _check_url_responds(repo.homepage, url_cache))
        deploy_cfg_ok = _has_deploy_config(repo.file_tree)
        if not (homepage_ok and deploy_cfg_ok):
            verdict = Verdict.MODERATE
            reason = f"{reason} (capped: deployment STRONG requires live homepage and deploy config)".strip()

    if claim.type == "feature" and verdict == Verdict.STRONG:
        has_strong_indep_ref = False
        for ref in evidence:
            if not ref.independent:
                continue
            path = _extract_ref_path(ref, repo)
            if path is not None and _is_weak_feature_path(path, stripped_text):
                continue
            has_strong_indep_ref = True
            break
        if not has_strong_indep_ref:
            verdict = Verdict.MODERATE
            reason = f"{reason} (capped: weak ref path cannot make feature STRONG)".strip()

    if claim.type != "technology" and verdict == Verdict.STRONG:
        if not any(ref.independent for ref in evidence):
            verdict = Verdict.MODERATE
            reason = f"{reason} (capped: non-technology STRONG requires independent=True evidence)".strip()

    if claim.type != "technology" and verdict == Verdict.STRONG:
        content_refs = [ref for ref in evidence if ref.kind in ("fetched_file", "readme", "file", "homepage")]
        if content_refs:
            key_terms = _extract_key_terms(stripped_text, project_name, repo.name)
            snippets = [_get_ref_snippet_for_check(ref, repo, fetched_files)[1] for ref in content_refs]
            if key_terms and not any(_snippet_has_key_term(sn, key_terms) for sn in snippets):
                ref_relevance = "low"
                verdict = Verdict.MODERATE
                reason = f"{reason} (capped: ref_relevance=low, snippet lacks claim key terms)".strip()
            else:
                ref_relevance = "ok"

    return ClaimResult(
        claim_text=claim.text,
        claim_type=claim.type,
        source_section=source_section,
        project_name=project_name,
        matched_repo=repo.full_name,
        suggested_repo=None,
        verdict=verdict,
        evidence=evidence,
        checked=checked,
        reason=reason,
        signals=signals,
        candidate_action=candidate_action,
        ref_relevance=ref_relevance,
        verification_answer=None,
        model=model_name,
    )


def _evaluate_project_batch(
    indexed_claims: list[tuple[int, Claim, str, bool]],
    source_section: str,
    project_name: str,
    repo: RepoEvidence,
    models_used: set[str],
    counters: dict[str, int],
    url_cache: dict[str, bool],
) -> dict[int, ClaimResult]:
    repo_payload = _build_repo_payload(repo)
    expected_indices = {idx for idx, _, _, _ in indexed_claims}
    claims_for_prompt = [(idx, claim, stripped) for idx, claim, stripped, _ in indexed_claims]

    try:
        fetched_files = _select_and_fetch_files(
            claims_for_prompt,
            repo,
            models_used,
            counters,
        )
        prompt = _build_pass2_prompt(claims_for_prompt, repo_payload, fetched_files)
        parsed, model_name = _call_pass2_and_validate(
            prompt,
            expected_indices,
            models_used,
            counters,
        )

        by_idx = {r.claim_index: r for r in parsed.results}
        out: dict[int, ClaimResult] = {}
        strong_to_verify: list[tuple[int, ClaimResult, str]] = []

        for idx, claim, stripped, had_authorship in indexed_claims:
            item = by_idx.get(idx)
            if item is None:
                signals = ["authorship claim cannot be verified from repo files"] if had_authorship else []
                out[idx] = ClaimResult(
                    claim_text=claim.text,
                    claim_type=claim.type,
                    source_section=source_section,
                    project_name=project_name,
                    matched_repo=repo.full_name,
                    verdict=Verdict.INSUFFICIENT_EVIDENCE,
                    evidence=[],
                    checked=[],
                    reason="checker failed",
                    signals=signals,
                    candidate_action="retry check or explain in a dispute",
                    model=model_name,
                )
            else:
                res = _finalize_claim_result(
                    claim,
                    stripped,
                    had_authorship,
                    source_section,
                    project_name,
                    repo,
                    item,
                    fetched_files,
                    model_name,
                    url_cache,
                )
                out[idx] = res
                if res.verdict == Verdict.STRONG and res.claim_type != "technology":
                    strong_to_verify.append((idx, res, stripped))

        if strong_to_verify:
            try:
                _verify_strong_non_tech_claims(
                    strong_to_verify,
                    repo,
                    fetched_files,
                    models_used,
                    counters,
                )
            except Exception:
                for _, res, _ in strong_to_verify:
                    res.verdict = Verdict.INSUFFICIENT_EVIDENCE
                    res.verification_answer = "checker failed"
                    res.reason = f"{res.reason} (downgraded: verification checker failed)".strip()

        return out
    except Exception:
        fallback_model = settings.gemini_model
        return {
            idx: ClaimResult(
                claim_text=claim.text,
                claim_type=claim.type,
                source_section=source_section,
                project_name=project_name,
                matched_repo=repo.full_name,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                evidence=[],
                checked=[],
                reason="checker failed",
                signals=(["authorship claim cannot be verified from repo files"] if had_authorship else []),
                candidate_action="retry check or explain in a dispute",
                model=fallback_model,
            )
            for idx, claim, _, had_authorship in indexed_claims
        }


def _evaluate_item_claims(
    claims: list[Claim],
    source_section: str,
    project_name: str,
    links: list[str],
    repo_map: dict[str, RepoEvidence],
    all_repos: list[RepoEvidence],
    models_used: set[str],
    counters: dict[str, int],
    url_cache: dict[str, bool],
) -> tuple[list[ClaimResult], str | None]:
    matched = _match_linked_repo(links, repo_map)
    if matched is None or matched.fetch_error is not None:
        current_model = settings.gemini_model
        results = []
        for claim in claims:
            _, had_auth = _strip_authorship_words(claim.text)
            signals = ["authorship claim cannot be verified from repo files"] if had_auth else []
            results.append(
                ClaimResult(
                    claim_text=claim.text,
                    claim_type=claim.type,
                    source_section=source_section,
                    project_name=project_name,
                    matched_repo=matched.full_name if matched else None,
                    verdict=Verdict.INSUFFICIENT_EVIDENCE,
                    evidence=[],
                    checked=[],
                    reason="no linked public repo",
                    signals=signals,
                    candidate_action="link a public repo or explain in a dispute",
                    model=current_model,
                )
            )
        return results, (matched.full_name if matched else None)

    ordered: dict[int, ClaimResult] = {}
    batch_claims: list[tuple[int, Claim, str, bool]] = []
    current_model = settings.gemini_model

    for idx, claim in enumerate(claims):
        stripped, had_auth = _strip_authorship_words(claim.text)
        mismatched = _find_mismatched_repo(claim, project_name, matched, all_repos)
        if mismatched is not None:
            signals = ["authorship claim cannot be verified from repo files"] if had_auth else []
            ordered[idx] = ClaimResult(
                claim_text=claim.text,
                claim_type=claim.type,
                source_section=source_section,
                project_name=project_name,
                matched_repo=matched.full_name,
                suggested_repo=mismatched.url,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                evidence=[],
                checked=[],
                reason="claim refers to a repo that is not linked",
                signals=signals,
                candidate_action=f"Link {mismatched.url}?",
                model=current_model,
            )
        else:
            batch_claims.append((idx, claim, stripped, had_auth))

    if batch_claims:
        batch_results = _evaluate_project_batch(
            batch_claims,
            source_section,
            project_name,
            matched,
            models_used,
            counters,
            url_cache,
        )
        ordered.update(batch_results)

    return [ordered[i] for i in range(len(claims))], matched.full_name


def _summarize_project(
    project_name: str,
    repo_name: str | None,
    results: list[ClaimResult],
    repo: RepoEvidence | None,
) -> tuple[ProjectSummary, EvidencedProject | None]:
    counts = {v.value: 0 for v in Verdict}
    strong_feature_claims = 0
    strong_indep_results: list[ClaimResult] = []
    moderate_feature_results: list[ClaimResult] = []

    for r in results:
        counts[r.verdict.value] = counts.get(r.verdict.value, 0) + 1
        if r.claim_type == "feature":
            if r.verdict == Verdict.STRONG:
                strong_feature_claims += 1
                stripped_text, _ = _strip_authorship_words(r.claim_text)
                has_valid_indep = False
                for ref in r.evidence:
                    if not ref.independent:
                        continue
                    path = _extract_ref_path(ref, repo) if repo else None
                    if path is not None and _is_weak_feature_path(path, stripped_text):
                        continue
                    has_valid_indep = True
                    break
                if has_valid_indep:
                    strong_indep_results.append(r)
            elif r.verdict == Verdict.MODERATE:
                moderate_feature_results.append(r)

    strong_indep_count = len(strong_indep_results)
    moderate_feature_count = len(moderate_feature_results)

    reasons: list[str] = []
    qualifying_claims: list[str] = []
    qualifying_refs: list[EvidenceRef] = []

    if strong_indep_count >= 1:
        reasons.append(f"{strong_indep_count} STRONG independent feature claim(s)")
        first_strong = strong_indep_results[0]
        stripped_text, _ = _strip_authorship_words(first_strong.claim_text)
        qualifying_claims.append(first_strong.claim_text)
        for ref in first_strong.evidence:
            if not ref.independent:
                continue
            path = _extract_ref_path(ref, repo) if repo else None
            if path is not None and _is_weak_feature_path(path, stripped_text):
                continue
            qualifying_refs.append(ref)
    if moderate_feature_count >= settings.min_moderate_features:
        reasons.append(f"{moderate_feature_count} MODERATE feature claims")
        if not qualifying_claims:
            for mr in moderate_feature_results[: settings.min_moderate_features]:
                qualifying_claims.append(mr.claim_text)
                qualifying_refs.extend(mr.evidence[:1])

    evidenced = bool(reasons)
    evidenced_reason = (
        "; ".join(reasons)
        if evidenced
        else f"0 STRONG independent feature claims and {moderate_feature_count}/{settings.min_moderate_features} MODERATE feature claims"
    )

    summary = ProjectSummary(
        project_name=project_name,
        repo=repo_name,
        counts=counts,
        strong_feature_claims=strong_feature_claims,
        strong_independent_features=strong_indep_count,
        moderate_feature_claims=moderate_feature_count,
        evidenced=evidenced,
        evidenced_reason=evidenced_reason,
    )
    evidenced_proj = (
        EvidencedProject(
            project_name=project_name,
            repo=repo_name,
            reason=evidenced_reason,
            qualifying_claims=qualifying_claims,
            qualifying_refs=qualifying_refs,
        )
        if evidenced
        else None
    )
    return summary, evidenced_proj


def check_resume(resume: ResumeProfile, github: GitHubProfile) -> EvidenceReport:
    repo_map: dict[str, RepoEvidence] = {}
    for repo in github.repos:
        repo_map[repo.full_name.lower()] = repo
        url_key = _parse_github_repo_key(repo.url)
        if url_key:
            repo_map[url_key] = repo

    results: list[ClaimResult] = []
    project_summaries: list[ProjectSummary] = []
    evidenced_projects: list[EvidencedProject] = []
    models_used: set[str] = set()
    counters = {"llm_calls": 0, "fetch_file_calls": 0}
    url_cache: dict[str, bool] = {}

    for exp in resume.experience:
        exp_results, _ = _evaluate_item_claims(
            exp.claims,
            "experience",
            exp.role,
            exp.links,
            repo_map,
            github.repos,
            models_used,
            counters,
            url_cache,
        )
        results.extend(exp_results)

    for proj in resume.projects:
        proj_results, repo_name = _evaluate_item_claims(
            proj.claims,
            "projects",
            proj.name,
            proj.links,
            repo_map,
            github.repos,
            models_used,
            counters,
            url_cache,
        )
        results.extend(proj_results)
        matched_repo = repo_map.get(repo_name.lower()) if repo_name else None
        summary, evidenced_proj = _summarize_project(proj.name, repo_name, proj_results, matched_repo)
        project_summaries.append(summary)
        if evidenced_proj is not None:
            evidenced_projects.append(evidenced_proj)

    result_models = {r.model for r in results if r.model}
    all_seen_models = models_used | result_models
    if len(all_seen_models) > 1:
        print(
            f"Warning: multiple models used during check_resume run: {sorted(all_seen_models)}",
            file=sys.stderr,
        )

    evidence_gate_passed = len(evidenced_projects) >= settings.min_projects_with_evidence

    return EvidenceReport(
        results=results,
        projects=project_summaries,
        evidenced_projects=evidenced_projects,
        evidence_gate_passed=evidence_gate_passed,
        llm_calls=counters["llm_calls"],
        fetch_file_calls=counters["fetch_file_calls"],
        model=settings.gemini_model,
        prompt_version=PROMPT_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
