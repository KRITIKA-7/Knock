import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

from backend.app.config import (
    BASE_DIR,
    extract_usable_grounding_ref,
    is_allowed_tree_file,
    is_weak_match,
    settings,
    split_fetched_detail,
)
from backend.app.schemas import (
    DroppedQuestion,
    EvidenceRef,
    EvidenceReport,
    FitMatch,
    FitResult,
    GitHubProfile,
    HardProblem,
    Question,
    QuestionSet,
    RepoEvidence,
    StartupProfile,
    Verdict,
)
from backend.app.services.github_analyzer import fetch_file
from backend.app.services.llm_client import call_llm, invalidate_llm_cache

PROMPT_VERSION = "v4"
MAX_CANDIDATES_PER_PAIR = 3
MAX_SELECTED_FILES_PER_PAIR = 3
MAX_CONTEXT_CHARS = 8192
MAX_FETCH_BYTES = 20000
FILE_CACHE_DIR = BASE_DIR / "samples" / ".cache"

FUNC_HDR_PATTERN = re.compile(
    r"^\s*(?:(?:public|protected|private|static|final|abstract|async|export|default)\s+)*(?:async\s+def|def|function)\b"
    r"|^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>\s*\{"
)
CLASS_HDR_PATTERN = re.compile(
    r"^\s*(?:(?:public|protected|private|abstract|final|export|default)\s+)*(?:class|interface|struct|impl|enum)\b"
)

FILE_SELECT_SYSTEM_PROMPT = (
    "You select source code files from a repository file tree most likely to contain "
    "concrete engineering design decisions relevant to a startup problem. Return ONLY valid JSON."
)

SYSTEM_PROMPT = (
    "You write grounded technical interview questions connecting verified repository design decisions "
    "to a startup's real engineering problems. Return ONLY valid JSON."
)

VERIFIER_SYSTEM_PROMPT = (
    "You strictly audit interview question grounding. Decide whether the provided code or artifact context "
    "shows a concrete design decision relevant to the startup problem. Return ONLY valid JSON."
)

FILE_SELECT_SCHEMA_DESC = """{
  "files": [string]
}"""

OUTPUT_SCHEMA_DESC = """{
  "questions": [
    {
      "id": "q1",
      "text": string,
      "project_name": string,
      "grounded_in": [
        {
          "kind": "fetched_file",
          "detail": string
        }
      ],
      "startup_problem_id": string,
      "time_limit_minutes": 15
    }
  ]
}"""

VERIFIER_SCHEMA_DESC = """{
  "verifications": [
    {
      "question_id": string,
      "verdict": "YES" | "PARTIAL" | "NO",
      "reason": string
    }
  ]
}"""


class _FileSelectOutput(BaseModel):
    files: list[str] = Field(default_factory=list)


class _LLMQuestionList(BaseModel):
    questions: list[Question] = Field(default_factory=list)


class _VerificationItem(BaseModel):
    question_id: str
    verdict: Literal["YES", "PARTIAL", "NO"]
    reason: str


class _VerificationList(BaseModel):
    verifications: list[_VerificationItem] = Field(default_factory=list)


@dataclass
class _UsableRef:
    ref: EvidenceRef
    file_path: str
    snippet: str
    expanded_context: str


@dataclass
class _UsablePair:
    match: FitMatch
    problem: HardProblem
    refs: list[_UsableRef]


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _extract_json_object(raw: str) -> str:
    text = _strip_code_fences(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _file_cache_path(full_name: str, path: str) -> Path:
    key = hashlib.sha256(f"{full_name}|{path}|{MAX_FETCH_BYTES}".encode("utf-8")).hexdigest()[:16]
    return FILE_CACHE_DIR / f"file_{key}.json"


def _fetch_file_cached(
    full_name: str,
    path: str,
    counters: dict[str, int],
) -> str | None:
    cache_path = _file_cache_path(full_name, path)
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "content" in payload:
                content = payload["content"]
                if isinstance(content, str) or content is None:
                    counters["cached_fetches"] += 1
                    return content
        except (OSError, ValueError):
            pass

    counters["fetch_count"] += 1
    try:
        content = fetch_file(full_name, path, max_bytes=MAX_FETCH_BYTES)
    except Exception:
        return None

    try:
        FILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"full_name": full_name, "path": path, "content": content}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass
    return content


def _find_target_line(lines: list[str], cited_snippet: str) -> int | None:
    needle_lines = [ln.strip() for ln in cited_snippet.splitlines() if ln.strip()]
    if not needle_lines:
        return None
    needle = needle_lines[0]
    for idx, ln in enumerate(lines):
        if needle in ln:
            return idx
    compact = re.sub(r"\s+", " ", needle)
    for idx, ln in enumerate(lines):
        if compact in re.sub(r"\s+", " ", ln.strip()):
            return idx
    return None


def _extract_braced_block(lines: list[str], hdr_idx: int) -> int | None:
    depth = 0
    started = False
    for idx in range(hdr_idx, len(lines)):
        line = lines[idx]
        in_quote: str | None = None
        escaped = False
        for ch in line:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_quote is not None:
                if ch == in_quote:
                    in_quote = None
                continue
            if ch in ("'", '"'):
                in_quote = ch
                continue
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}" and started:
                depth -= 1
                if depth == 0:
                    return idx
    return None


def _extract_indented_block(lines: list[str], hdr_idx: int) -> int:
    hdr_line = lines[hdr_idx]
    hdr_indent = len(hdr_line) - len(hdr_line.lstrip())
    end_idx = hdr_idx
    for idx in range(hdr_idx + 1, len(lines)):
        raw = lines[idx]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent <= hdr_indent:
            break
        end_idx = idx
    return end_idx


def _extract_enclosing_block(lines: list[str], target_idx: int, max_chars: int) -> str | None:
    func_hdr: int | None = None
    class_hdr: int | None = None
    low_bound = max(0, target_idx - 250)

    for idx in range(target_idx, low_bound - 1, -1):
        line = lines[idx]
        if func_hdr is None and FUNC_HDR_PATTERN.match(line):
            func_hdr = idx
        if class_hdr is None and CLASS_HDR_PATTERN.match(line):
            class_hdr = idx
        if func_hdr is not None and class_hdr is not None:
            break

    for hdr_idx in (func_hdr, class_hdr):
        if hdr_idx is None:
            continue
        lookahead = "\n".join(lines[hdr_idx : min(len(lines), hdr_idx + 3)])
        if "{" in lookahead:
            end_idx = _extract_braced_block(lines, hdr_idx)
        elif lines[hdr_idx].rstrip().endswith(":"):
            end_idx = _extract_indented_block(lines, hdr_idx)
        else:
            continue
        if end_idx is None or end_idx < target_idx:
            continue
        block = "\n".join(lines[hdr_idx : end_idx + 1])
        if len(block) <= max_chars:
            return block

    return None


def _window_around_line(lines: list[str], target_idx: int, max_chars: int) -> str:
    lo = target_idx
    hi = target_idx
    total_len = len(lines[target_idx])
    if total_len > max_chars:
        return lines[target_idx][:max_chars]

    while True:
        grew = False
        if hi + 1 < len(lines) and total_len + 1 + len(lines[hi + 1]) <= max_chars:
            hi += 1
            total_len += 1 + len(lines[hi])
            grew = True
        if lo - 1 >= 0 and total_len + 1 + len(lines[lo - 1]) <= max_chars:
            lo -= 1
            total_len += 1 + len(lines[lo])
            grew = True
        if not grew:
            break
    return "\n".join(lines[lo : hi + 1])


def _extract_context_window(content: str, cited_snippet: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(content) <= max_chars:
        return content
    lines = content.splitlines()
    target_idx = _find_target_line(lines, cited_snippet)
    if target_idx is not None:
        block = _extract_enclosing_block(lines, target_idx, max_chars)
        if block is not None:
            return block
        return _window_around_line(lines, target_idx, max_chars)
    lines_cap: list[str] = []
    total = 0
    for ln in lines:
        add_len = len(ln) + (1 if lines_cap else 0)
        if total + add_len > max_chars:
            break
        lines_cap.append(ln)
        total += add_len
    return "\n".join(lines_cap) if lines_cap else content[:max_chars]


def _pick_anchor_line(content: str, problem_text: str = "") -> tuple[str, int]:
    lines = content.splitlines()
    keywords = {
        tok
        for tok in re.findall(r"[a-z0-9_]{3,}", problem_text.lower())
        if tok not in {"the", "and", "for", "with", "from", "that", "this", "into", "under", "when"}
    }
    best_idx: int | None = None
    best_score = 0
    if keywords:
        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            if not stripped or stripped.startswith(("#", "//", "/*", "*")):
                continue
            low = stripped.lower()
            score = sum(1 for kw in keywords if kw in low)
            if score > 0 and (FUNC_HDR_PATTERN.match(raw) or CLASS_HDR_PATTERN.match(raw)):
                score += 2
            if score > best_score:
                best_score = score
                best_idx = idx
    if best_idx is not None:
        return lines[best_idx].strip()[:120], best_idx

    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped and (FUNC_HDR_PATTERN.match(raw) or CLASS_HDR_PATTERN.match(raw)):
            return stripped[:120], idx
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped and stripped not in ("<?php", "{", "}", "[", "]") and not stripped.startswith(("#", "//", "/*", "*")):
            return stripped[:120], idx
    return (lines[0].strip()[:120], 0) if lines else ("source file", 0)


def _resolve_repo_full_name(
    match: FitMatch,
    report: EvidenceReport | None,
) -> str | None:
    if match.matched_repo:
        return match.matched_repo
    if report is None:
        return None
    proj_key = match.project_name.strip().lower()
    for ps in report.projects:
        if ps.project_name.strip().lower() == proj_key and ps.repo:
            return ps.repo
    for r in report.results:
        if r.project_name.strip().lower() == proj_key and r.matched_repo:
            return r.matched_repo
    return None


def _find_repo_in_github(
    repo_full_name: str | None,
    match: FitMatch,
    github: GitHubProfile | None,
) -> RepoEvidence | None:
    if github is None:
        return None
    if repo_full_name:
        target = repo_full_name.strip().lower()
        for repo in github.repos:
            if repo.full_name.strip().lower() == target:
                return repo
    proj_key = match.project_name.strip().lower()
    for repo in github.repos:
        if repo.name.strip().lower() == proj_key:
            return repo
    return None


def _build_expanded_context(
    repo_full_name: str | None,
    file_path: str,
    snippet: str,
    counters: dict[str, int],
) -> str:
    if repo_full_name:
        content = _fetch_file_cached(repo_full_name, file_path, counters)
        if content:
            return _extract_context_window(content, snippet, max_chars=MAX_CONTEXT_CHARS)
    return snippet[:MAX_CONTEXT_CHARS]


def _select_and_fetch_pair_files(
    pair: _UsablePair,
    repo: RepoEvidence | None,
    counters: dict[str, int],
) -> list[_UsableRef]:
    if repo is None or not repo.file_tree:
        return []

    prompt = f"""Select up to {MAX_SELECTED_FILES_PER_PAIR} exact file paths from file_tree most likely to hold a relevant design decision (error handling, retries, caching, thresholds, data flow) for this startup problem.

Return ONLY JSON matching:
{FILE_SELECT_SCHEMA_DESC}

Rules:
- Choose only exact strings present in file_tree.
- Pick implementation code modules, controllers, pipelines, models, or inference/export scripts.
- Do NOT choose lockfiles, data files, binary/image/weight files, locale/translation files, config/manifest files, docs, or test files.

Input JSON:
{json.dumps({"project_name": pair.match.project_name, "startup_problem": pair.problem.model_dump(), "file_tree": repo.file_tree}, ensure_ascii=False)}"""

    counters["llm_calls"] += 1
    raw = call_llm(prompt, system=FILE_SELECT_SYSTEM_PROMPT, temperature=0)
    try:
        selection = _FileSelectOutput.model_validate_json(_extract_json_object(raw))
    except (ValidationError, ValueError):
        invalidate_llm_cache(prompt, system=FILE_SELECT_SYSTEM_PROMPT, temperature=0)
        return []

    allowed_tree = set(repo.file_tree)
    existing_paths = {r.file_path for r in pair.refs}
    chosen_paths: list[str] = []
    seen_chosen: set[str] = set()

    for path in selection.files:
        clean_path = path.strip()
        if (
            clean_path in allowed_tree
            and clean_path not in seen_chosen
            and clean_path not in existing_paths
            and is_allowed_tree_file(clean_path)
        ):
            seen_chosen.add(clean_path)
            chosen_paths.append(clean_path)
            if len(chosen_paths) >= MAX_SELECTED_FILES_PER_PAIR:
                break

    problem_text = f"{pair.problem.title} {pair.problem.description}"
    added_refs: list[_UsableRef] = []
    for path in chosen_paths:
        content = _fetch_file_cached(repo.full_name, path, counters)
        if not content:
            continue
        anchor, target_idx = _pick_anchor_line(content, problem_text)
        if len(content) <= MAX_CONTEXT_CHARS:
            expanded = content
        else:
            lines = content.splitlines()
            expanded = _window_around_line(lines, target_idx, MAX_CONTEXT_CHARS)
        ref = EvidenceRef(kind="fetched_file", detail=f"{path}: {anchor}", independent=True)
        added_refs.append(
            _UsableRef(
                ref=ref,
                file_path=path,
                snippet=anchor,
                expanded_context=expanded,
            )
        )
    return added_refs


def _collect_usable_pairs(
    fit_result: FitResult,
    startup: StartupProfile,
    report: EvidenceReport | None,
    github: GitHubProfile | None,
    counters: dict[str, int],
) -> tuple[list[_UsablePair], int]:
    problem_map = {p.id: p for p in startup.hard_problems}
    ordered_matches = sorted(
        [m for m in fit_result.matches if m.relation in settings.weak_match_relations],
        key=lambda m: 0 if m.relation == "direct" else 1,
    )

    candidate_problem_matches = [
        m for m in ordered_matches if m.startup_problem_id and m.startup_problem_id in problem_map
    ]

    usable_pairs: list[_UsablePair] = []
    seen_pairs: set[tuple[str, str]] = set()
    for match in candidate_problem_matches:
        prob = problem_map[match.startup_problem_id]  # type: ignore[index]
        pair_key = (match.project_name.strip().lower(), prob.id)
        if pair_key in seen_pairs:
            continue
        context = f"{match.explanation} {prob.title} {prob.description}"
        repo_full_name = _resolve_repo_full_name(match, report)
        repo = _find_repo_in_github(repo_full_name, match, github)
        resolved_full_name = repo.full_name if repo is not None else repo_full_name

        usable_refs: list[_UsableRef] = []
        seen_paths: set[str] = set()
        for ref in match.evidence_refs:
            extracted = extract_usable_grounding_ref(ref, context)
            if extracted is not None and extracted.file_path not in seen_paths:
                seen_paths.add(extracted.file_path)
                expanded = _build_expanded_context(
                    resolved_full_name,
                    extracted.file_path,
                    extracted.snippet,
                    counters,
                )
                usable_refs.append(
                    _UsableRef(
                        ref=extracted.ref,
                        file_path=extracted.file_path,
                        snippet=extracted.snippet,
                        expanded_context=expanded,
                    )
                )

        if usable_refs:
            seen_pairs.add(pair_key)
            pair = _UsablePair(match=match, problem=prob, refs=usable_refs)
            extra_refs = _select_and_fetch_pair_files(pair, repo, counters)
            pair.refs.extend(extra_refs)
            usable_pairs.append(pair)

    return usable_pairs, len(candidate_problem_matches)


def _closest_work_summary(
    fit_result: FitResult,
    report: EvidenceReport | None,
) -> list[str]:
    summaries: list[str] = []
    seen: set[str] = set()

    for m in fit_result.matches:
        target = m.startup_problem_id or "tech_stack"
        entry = f"{m.project_name} ({m.relation} -> {target}): {m.explanation}"
        if entry not in seen:
            seen.add(entry)
            summaries.append(entry)

    if fit_result.tech_overlap_sources:
        overlaps = ", ".join(f"{k} via {v}" for k, v in fit_result.tech_overlap_sources.items())
        summaries.append(f"Verified stack overlap: {overlaps}")

    if not summaries and report is not None:
        by_project: dict[str, list[str]] = {}
        for r in report.results:
            if r.verdict in (Verdict.STRONG, Verdict.MODERATE):
                by_project.setdefault(r.project_name, []).append(r.claim_text)
        for proj, claims in list(by_project.items())[:3]:
            summaries.append(f"{proj}: {'; '.join(claims[:2])}")

    return summaries[:5]


def _extract_detail_terms(detail: str, context: str = "") -> list[str]:
    terms: list[str] = []
    for src in (detail, context):
        for tok in re.findall(r"[A-Za-z0-9_.\-]{3,}", src):
            t = tok.lower().strip(".-_")
            if len(t) >= 3:
                terms.append(t)
    return terms


def _looks_like_bare_filename_question(text: str) -> bool:
    cleaned = re.sub(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,8}", "", text).strip()
    words = re.findall(r"[A-Za-z]{2,}", cleaned)
    return len(words) < 8


def _build_generation_prompt(pairs: list[_UsablePair], startup: StartupProfile) -> str:
    pair_payload = [
        {
            "project_name": pair.match.project_name,
            "relation": pair.match.relation,
            "startup_problem": pair.problem.model_dump(),
            "match_explanation": pair.match.explanation,
            "approved_refs": [
                {
                    "kind": r.ref.kind,
                    "detail": r.ref.detail,
                    "file_path": r.file_path,
                    "context": r.expanded_context,
                }
                for r in pair.refs
            ],
        }
        for pair in pairs
    ]
    max_candidates = MAX_CANDIDATES_PER_PAIR * len(pairs)
    return f"""Generate candidate technical questions for ONLY the verified project-problem pairs below.
Generate up to {MAX_CANDIDATES_PER_PAIR} distinct candidate questions PER pair (up to {max_candidates} total) using distinct engineering angles: failure case, scaling under load, and alternative design trade-offs.

Return ONLY JSON matching this schema:
{OUTPUT_SCHEMA_DESC}

Rules:
- Sequential ids "q1", "q2", ...
- Every candidate MUST use one listed (project_name, startup_problem.id) pair. Never invent unlisted problems.
- When multiple approved_refs exist for a pair, ground different candidates in different files/functions where possible.
- Use each approved ref's full "context" to describe the concrete implementation decision, threshold, error handling, caching, or data flow in plain words (do NOT merely cite a file path or symbol without explaining what it does).
- Connect that specific verified design detail to the paired startup_problem: ask where their current design breaks under that constraint and what they would change.
- Every item in "grounded_in" MUST copy an exact ("kind", "detail") pair from that pair's approved_refs list.
- Answerable in 150 to 300 words (set time_limit_minutes to 15 or 20).

Input JSON:
{json.dumps({"startup": {"domain": startup.domain, "tech_stack": startup.tech_stack}, "usable_pairs": pair_payload}, ensure_ascii=False)}"""


def _validate_generated(parsed: _LLMQuestionList, pairs: list[_UsablePair]) -> list[Question]:
    if not parsed.questions:
        raise ValueError("Expected at least 1 candidate question")
    max_allowed = MAX_CANDIDATES_PER_PAIR * len(pairs)
    if len(parsed.questions) > max_allowed:
        raise ValueError(
            f"Expected at most {max_allowed} candidate question(s), got {len(parsed.questions)}"
        )

    pair_lookup: dict[tuple[str, str], _UsablePair] = {
        (p.match.project_name.strip().lower(), p.problem.id): p for p in pairs
    }

    validated: list[Question] = []
    seen_texts: set[str] = set()
    for idx, q in enumerate(parsed.questions, start=1):
        pair_key = (q.project_name.strip().lower(), q.startup_problem_id)
        pair = pair_lookup.get(pair_key)
        if pair is None:
            raise ValueError(
                f"Question {q.id} targets unlisted pair ({q.project_name!r}, {q.startup_problem_id!r})"
            )

        if not q.grounded_in:
            raise ValueError(f"Question {q.id} has empty grounded_in")

        ref_map = {(r.ref.kind, r.ref.detail): r for r in pair.refs}
        path_map = {r.file_path: r for r in pair.refs}
        normalized_refs: list[EvidenceRef] = []
        detail_terms: list[str] = []
        seen_context_details: set[str] = set()
        contexts: list[str] = []

        for ref in q.grounded_in:
            matched = ref_map.get((ref.kind, ref.detail))
            if matched is None:
                split = split_fetched_detail(ref.detail)
                candidate_path = split[0] if split else ref.detail.strip()
                matched = path_map.get(candidate_path)
            if matched is None:
                raise ValueError(
                    f"Question {q.id} cites unapproved grounded_in ref: ({ref.kind}, {ref.detail!r})"
                )
            normalized_refs.append(matched.ref.model_copy())
            detail_terms.extend(_extract_detail_terms(matched.ref.detail, matched.expanded_context))
            if matched.ref.detail not in seen_context_details:
                seen_context_details.add(matched.ref.detail)
                contexts.append(matched.expanded_context)

        clean_text = q.text.strip()
        if _looks_like_bare_filename_question(clean_text):
            raise ValueError(
                f"Question {q.id} must describe the design detail in plain words, not just name a file"
            )

        q_low = clean_text.lower()
        proj_low = pair.match.project_name.lower()
        mentions_project = proj_low in q_low or any(
            w in q_low for w in re.findall(r"[a-z0-9]{4,}", proj_low)
        )
        mentions_term = any(term in q_low for term in detail_terms)
        if not (mentions_project or mentions_term):
            raise ValueError(
                f"Question {q.id} does not mention project_name or concrete detail terms"
            )

        if q_low in seen_texts:
            continue
        seen_texts.add(q_low)

        combined_context = "\n\n".join(contexts)
        validated.append(
            Question(
                id=f"c{idx}",
                text=clean_text,
                project_name=pair.match.project_name,
                grounded_in=normalized_refs,
                startup_problem_id=pair.problem.id,
                time_limit_minutes=q.time_limit_minutes,
                context_chars=len(combined_context),
                context_preview=combined_context[:300],
            )
        )

    if not validated:
        raise ValueError("No distinct valid candidate questions generated")
    return validated


def _question_files(q: Question) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for ref in q.grounded_in:
        split = split_fetched_detail(ref.detail)
        fpath = split[0] if split else ref.detail.strip()
        if fpath and fpath not in seen:
            seen.add(fpath)
            files.append(fpath)
    return files


def _order_diverse_candidates(yes_items: list[tuple[Question, str]]) -> list[tuple[Question, str]]:
    buckets: dict[str, list[tuple[Question, str]]] = {}
    for item in yes_items:
        buckets.setdefault(item[0].startup_problem_id, []).append(item)
    ordered: list[tuple[Question, str]] = []
    keys = list(buckets.keys())
    idx = 0
    while len(ordered) < len(yes_items):
        for k in keys:
            if idx < len(buckets[k]):
                ordered.append(buckets[k][idx])
        idx += 1
    return ordered


def _verify_batch(
    candidates: list[Question],
    pairs: list[_UsablePair],
    startup: StartupProfile,
    verifier_model: str,
) -> tuple[list[Question], list[DroppedQuestion], list[str]]:
    problem_map = {p.id: p for p in startup.hard_problems}
    ref_context_map: dict[tuple[str, str, str], _UsableRef] = {}
    for pair in pairs:
        proj_key = pair.match.project_name.strip().lower()
        for r in pair.refs:
            ref_context_map[(proj_key, r.ref.kind, r.ref.detail)] = r

    items = []
    for q in candidates:
        prob = problem_map.get(q.startup_problem_id)
        proj_key = q.project_name.strip().lower()
        cited = []
        contexts = []
        paths = []
        for ref in q.grounded_in:
            usable = ref_context_map.get((proj_key, ref.kind, ref.detail))
            if usable is not None:
                paths.append(usable.file_path)
                contexts.append(usable.expanded_context)
                cited.append({"file_path": usable.file_path, "context": usable.expanded_context})
            else:
                split = split_fetched_detail(ref.detail)
                fpath = split[0] if split else ref.detail
                snip = split[1] if split else ""
                paths.append(fpath)
                if snip:
                    contexts.append(snip)
                cited.append({"file_path": fpath, "context": snip})

        combined_context = "\n\n".join(c for c in contexts if c)
        items.append(
            {
                "question_id": q.id,
                "question_text": q.text,
                "file_path": "; ".join(paths),
                "cited_snippet": combined_context,
                "cited_evidence": cited,
                "startup_problem": prob.model_dump() if prob else {"id": q.startup_problem_id},
            }
        )

    prompt = f"""For each candidate item below, evaluate:
Does this snippet/context show a design decision relevant to this startup problem? Answer YES, PARTIAL or NO plus one sentence.

Return ONLY JSON matching this schema:
{VERIFIER_SCHEMA_DESC}

Items:
{json.dumps(items, ensure_ascii=False)}"""

    raw = call_llm(prompt, system=VERIFIER_SYSTEM_PROMPT, temperature=0, model=verifier_model)
    try:
        parsed = _VerificationList.model_validate_json(_extract_json_object(raw))
        verdict_map = {v.question_id: v for v in parsed.verifications}
    except (ValidationError, ValueError):
        invalidate_llm_cache(prompt, system=VERIFIER_SYSTEM_PROMPT, temperature=0, model=verifier_model)
        dropped_all = [
            DroppedQuestion(
                id=q.id,
                text=q.text,
                project_name=q.project_name,
                startup_problem_id=q.startup_problem_id,
                cited_files=_question_files(q),
                verdict="INVALID",
                reason="Verifier returned unparseable JSON response.",
            )
            for q in candidates
        ]
        reasons = [
            f"{d.id} ({d.project_name} -> {d.startup_problem_id}): INVALID - {d.reason}"
            for d in dropped_all
        ]
        return [], dropped_all, reasons

    yes_items: list[tuple[Question, str]] = []
    dropped: list[DroppedQuestion] = []
    dropped_reasons: list[str] = []

    for q in candidates:
        q_files = _question_files(q)
        v = verdict_map.get(q.id)
        if v is None:
            reason = "No verification verdict returned."
            dropped.append(
                DroppedQuestion(
                    id=q.id,
                    text=q.text,
                    project_name=q.project_name,
                    startup_problem_id=q.startup_problem_id,
                    cited_files=q_files,
                    verdict="MISSING",
                    reason=reason,
                )
            )
            dropped_reasons.append(
                f"{q.id} ({q.project_name} -> {q.startup_problem_id}): MISSING - {reason}"
            )
            continue

        reason = v.reason.strip()
        if v.verdict == "YES":
            yes_items.append((q, reason))
        else:
            dropped.append(
                DroppedQuestion(
                    id=q.id,
                    text=q.text,
                    project_name=q.project_name,
                    startup_problem_id=q.startup_problem_id,
                    cited_files=q_files,
                    verdict=v.verdict,
                    reason=reason,
                )
            )
            dropped_reasons.append(
                f"{q.id} ({q.project_name} -> {q.startup_problem_id}): {v.verdict} - {reason}"
            )

    kept: list[Question] = []
    used_files: set[str] = set()
    max_q = max(1, settings.max_questions)

    for q, reason in _order_diverse_candidates(yes_items):
        q_files = _question_files(q)
        if any(fp in used_files for fp in q_files):
            dup_reason = "Dropped YES duplicate: at most one question per cited file."
            dropped.append(
                DroppedQuestion(
                    id=q.id,
                    text=q.text,
                    project_name=q.project_name,
                    startup_problem_id=q.startup_problem_id,
                    cited_files=q_files,
                    verdict="YES (DROPPED_DUPLICATE_FILE)",
                    reason=f"{dup_reason} Verifier: {reason}",
                )
            )
            dropped_reasons.append(
                f"{q.id} ({q.project_name} -> {q.startup_problem_id}): YES (duplicate file) - {reason}"
            )
            continue

        if len(kept) >= max_q:
            cap_reason = f"Dropped YES overflow: capped at max_questions ({max_q})."
            dropped.append(
                DroppedQuestion(
                    id=q.id,
                    text=q.text,
                    project_name=q.project_name,
                    startup_problem_id=q.startup_problem_id,
                    cited_files=q_files,
                    verdict="YES (DROPPED_MAX_CAP)",
                    reason=f"{cap_reason} Verifier: {reason}",
                )
            )
            dropped_reasons.append(
                f"{q.id} ({q.project_name} -> {q.startup_problem_id}): YES (max_questions cap) - {reason}"
            )
            continue

        used_files.update(q_files)
        kept.append(
            q.model_copy(
                update={
                    "id": f"q{len(kept) + 1}",
                    "verification_verdict": "YES",
                    "verification_reason": reason,
                }
            )
        )

    return kept, dropped, dropped_reasons


def generate_questions(
    fit_result: FitResult,
    startup: StartupProfile,
    report: EvidenceReport | None = None,
    github: GitHubProfile | None = None,
) -> QuestionSet:
    verifier_model = settings.VERIFIER_MODEL
    valid_problem_ids = {p.id for p in startup.hard_problems}
    weak_by_rule = is_weak_match(fit_result.matches, valid_problem_ids=valid_problem_ids)
    closest = _closest_work_summary(fit_result, report)

    if weak_by_rule:
        return QuestionSet(
            startup_slug=startup.slug,
            questions=[],
            weak_match=True,
            reason=(
                f"No direct or adjacent matches connect verified work to {startup.name}'s hard problems; "
                "try a startup aligned with your verified projects."
            ),
            closest_verified_work=closest,
            candidates_generated=0,
            questions_verified=0,
            questions_dropped=0,
            dropped_questions=[],
            dropped_verdict_reasons=[],
            fetch_count=0,
            cached_fetches=0,
            llm_calls=0,
            model=settings.gemini_model,
            verifier_model=verifier_model,
            prompt_version=PROMPT_VERSION,
        )

    counters = {"llm_calls": 0, "fetch_count": 0, "cached_fetches": 0}
    usable_pairs, total_problem_matches = _collect_usable_pairs(
        fit_result, startup, report, github, counters
    )

    if not usable_pairs:
        if total_problem_matches == 0:
            reason = (
                f"No direct or adjacent matches connect verified work to {startup.name}'s hard problems; "
                "try a startup aligned with your verified projects."
            )
        else:
            reason = (
                f"Matched projects for {startup.name} relied only on disallowed grounding "
                "(binary, patch/diff, lockfile, locale, config/manifest, or one-line constants/assignments); "
                "try a startup with stronger code-level match evidence."
            )
        return QuestionSet(
            startup_slug=startup.slug,
            questions=[],
            weak_match=True,
            reason=reason,
            closest_verified_work=closest,
            candidates_generated=0,
            questions_verified=0,
            questions_dropped=0,
            dropped_questions=[],
            dropped_verdict_reasons=[],
            fetch_count=counters["fetch_count"],
            cached_fetches=counters["cached_fetches"],
            llm_calls=counters["llm_calls"],
            model=settings.gemini_model,
            verifier_model=verifier_model,
            prompt_version=PROMPT_VERSION,
        )

    prompt = _build_generation_prompt(usable_pairs, startup)
    counters["llm_calls"] += 1
    raw = call_llm(prompt, system=SYSTEM_PROMPT, temperature=0)

    try:
        parsed = _LLMQuestionList.model_validate_json(_strip_code_fences(raw))
        candidates = _validate_generated(parsed, usable_pairs)
    except (ValidationError, ValueError) as first_err:
        invalidate_llm_cache(prompt, system=SYSTEM_PROMPT, temperature=0)
        retry_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"Validation failed with error:\n{first_err}\n\n"
                    "Fix all errors and return ONLY valid JSON matching the exact schema."
                ),
            },
        ]
        counters["llm_calls"] += 1
        retry_raw = call_llm(retry_messages, system=SYSTEM_PROMPT, temperature=0)
        try:
            parsed_retry = _LLMQuestionList.model_validate_json(_strip_code_fences(retry_raw))
            candidates = _validate_generated(parsed_retry, usable_pairs)
        except (ValidationError, ValueError) as second_err:
            invalidate_llm_cache(retry_messages, system=SYSTEM_PROMPT, temperature=0)
            raise ValueError(
                f"Failed to generate valid questions after retry: {second_err}"
            ) from second_err

    counters["llm_calls"] += 1
    verified_questions, dropped_items, dropped_reasons = _verify_batch(
        candidates, usable_pairs, startup, verifier_model
    )
    cand_count = len(candidates)
    q_verified = len(verified_questions)
    q_dropped = len(dropped_items)

    if q_verified < settings.min_verified_questions:
        if q_verified == 0:
            reason = (
                f"No generated candidate questions received a YES verification verdict for {startup.name}; "
                "try another startup."
            )
        else:
            reason = (
                f"Only {q_verified} question(s) received a YES verification verdict for {startup.name} "
                f"(minimum required: {settings.min_verified_questions}); try another startup."
            )
        return QuestionSet(
            startup_slug=startup.slug,
            questions=[],
            weak_match=True,
            reason=reason,
            closest_verified_work=closest,
            candidates_generated=cand_count,
            questions_verified=q_verified,
            questions_dropped=q_dropped,
            dropped_questions=dropped_items,
            dropped_verdict_reasons=dropped_reasons,
            fetch_count=counters["fetch_count"],
            cached_fetches=counters["cached_fetches"],
            llm_calls=counters["llm_calls"],
            model=settings.gemini_model,
            verifier_model=verifier_model,
            prompt_version=PROMPT_VERSION,
        )

    return QuestionSet(
        startup_slug=startup.slug,
        questions=verified_questions,
        weak_match=False,
        reason=None,
        closest_verified_work=[],
        candidates_generated=cand_count,
        questions_verified=q_verified,
        questions_dropped=q_dropped,
        dropped_questions=dropped_items,
        dropped_verdict_reasons=dropped_reasons,
        fetch_count=counters["fetch_count"],
        cached_fetches=counters["cached_fetches"],
        llm_calls=counters["llm_calls"],
        model=settings.gemini_model,
        verifier_model=verifier_model,
        prompt_version=PROMPT_VERSION,
    )
