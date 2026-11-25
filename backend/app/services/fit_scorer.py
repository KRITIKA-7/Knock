import json
import re
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError

from backend.app.config import is_weak_match, settings
from backend.app.schemas import (
    EvidenceRef,
    EvidenceReport,
    FitMatch,
    FitResult,
    RelationType,
    StartupProfile,
    Verdict,
)
from backend.app.services.llm_client import call_llm, invalidate_llm_cache

PROMPT_VERSION = "v2"

BUILTIN_TECHNOLOGIES = {
    "sqlite",
    "sqlite3",
    "websockets",
    "websocket",
    "webrtc",
    "sse",
    "http",
    "https",
    "tcp",
    "udp",
    "sql",
    "html",
    "css",
    "json",
    "svg",
    "xml",
    "asyncio",
    "threading",
    "multiprocessing",
    "indexeddb",
    "localstorage",
    "sessionstorage",
    "wasm",
    "webassembly",
    "webgl",
    "canvas",
    "service worker",
    "service workers",
    "web workers",
    "posix",
    "uart",
    "spi",
    "i2c",
    "can-fd",
    "pwm",
    "adc",
    "dma",
    "gpio",
    "ble",
}

SYSTEM_PROMPT = (
    "You map verified candidate repository claims to a startup's engineering problems and tech stack. "
    "Do not compute or return numeric scores. Return ONLY valid JSON."
)

OUTPUT_SCHEMA_DESC = """{
  "matches": [
    {
      "project_name": string,
      "startup_problem_id": string | null,
      "relation": "direct" | "adjacent" | "transferable",
      "evidence_refs": [
        {
          "kind": "dependency" | "language" | "file" | "readme" | "commit_stat" | "homepage" | "fetched_file",
          "detail": string
        }
      ],
      "explanation": string
    }
  ]
}"""


class _LLMFitMatch(BaseModel):
    project_name: str
    startup_problem_id: str | None = None
    relation: RelationType
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    explanation: str


class _LLMFitMatchList(BaseModel):
    matches: list[_LLMFitMatch] = Field(default_factory=list)


@dataclass
class _CodeEvidenceItem:
    project_name: str
    kind: str
    detail: str
    path: str
    weight: float


@dataclass
class _ReportIndexes:
    project_canon: dict[str, str]
    project_repo: dict[str, str]
    ref_lookup: dict[tuple[str, str, str], EvidenceRef]
    ref_strength: dict[tuple[str, str, str], float]
    dep_strength_by_project: dict[str, dict[str, float]]
    dep_strength_global: dict[str, tuple[float, str]]
    lang_strength_by_project: dict[str, dict[str, float]]
    lang_strength_global: dict[str, tuple[float, str]]
    code_refs_by_project: dict[str, list[_CodeEvidenceItem]]
    code_refs_global: list[_CodeEvidenceItem]


def _clamp01(val: float) -> float:
    return round(max(0.0, min(1.0, val)), 4)


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


def _verdict_weight(verdict: Verdict) -> float:
    if verdict == Verdict.STRONG:
        return settings.fit_strong_weight
    if verdict == Verdict.MODERATE:
        return settings.fit_moderate_weight
    return 0.0


def _relation_weight(relation: str) -> float:
    if relation == "direct":
        return settings.fit_relation_direct_weight
    if relation == "adjacent":
        return settings.fit_relation_adjacent_weight
    if relation == "transferable":
        return settings.fit_relation_transferable_weight
    return 0.0


def _is_builtin_tech(tech: str) -> bool:
    return tech.strip().lower() in BUILTIN_TECHNOLOGIES


def _term_in_text(term: str, text: str) -> bool:
    t_low = term.strip().lower()
    if not t_low or not text:
        return False
    prefix = r"(?<![a-z0-9])" if t_low[0].isalnum() else ""
    suffix = r"(?![a-z0-9])" if t_low[-1].isalnum() else ""
    return bool(re.search(prefix + re.escape(t_low) + suffix, text.lower()))


def _extract_ref_path(kind: str, detail: str) -> str:
    if kind == "fetched_file":
        if ": " in detail:
            return detail.split(": ", 1)[0].strip()
        return detail.split(":", 1)[0].strip()
    return detail.strip()


def _collect_verified_claims(report: EvidenceReport) -> list[dict]:
    items: list[dict] = []
    for r in report.results:
        if r.verdict not in (Verdict.STRONG, Verdict.MODERATE):
            continue
        if not r.evidence:
            continue
        entry: dict = {
            "project_name": r.project_name,
            "verdict": r.verdict.value,
            "claim_text": r.claim_text,
            "claim_type": r.claim_type,
            "evidence": [
                {"kind": ref.kind, "detail": ref.detail, "independent": ref.independent}
                for ref in r.evidence
            ],
        }
        if r.verification_answer:
            entry["verified_snippet_summary"] = r.verification_answer
        items.append(entry)
    return items


def _build_report_indexes(report: EvidenceReport) -> _ReportIndexes:
    project_canon: dict[str, str] = {}
    project_repo: dict[str, str] = {}
    ref_lookup: dict[tuple[str, str, str], EvidenceRef] = {}
    ref_strength: dict[tuple[str, str, str], float] = {}
    dep_strength_by_project: dict[str, dict[str, float]] = {}
    dep_strength_global: dict[str, tuple[float, str]] = {}
    lang_strength_by_project: dict[str, dict[str, float]] = {}
    lang_strength_global: dict[str, tuple[float, str]] = {}
    code_refs_by_project: dict[str, list[_CodeEvidenceItem]] = {}
    code_refs_global: list[_CodeEvidenceItem] = []

    for ps in report.projects:
        if ps.repo:
            project_repo.setdefault(ps.project_name.strip().lower(), ps.repo)

    for r in report.results:
        if r.verdict not in (Verdict.STRONG, Verdict.MODERATE):
            continue
        if not r.evidence:
            continue
        proj_key = r.project_name.strip().lower()
        canon_proj = project_canon.setdefault(proj_key, r.project_name)
        if r.matched_repo:
            project_repo.setdefault(proj_key, r.matched_repo)
        weight = _verdict_weight(r.verdict)
        proj_deps = dep_strength_by_project.setdefault(proj_key, {})
        proj_langs = lang_strength_by_project.setdefault(proj_key, {})
        proj_code_refs = code_refs_by_project.setdefault(proj_key, [])

        for ref in r.evidence:
            key = (proj_key, ref.kind, ref.detail)
            ref_lookup.setdefault(key, ref)
            if weight > ref_strength.get(key, 0.0):
                ref_strength[key] = weight
            if ref.kind == "dependency":
                dep_key = ref.detail.strip().lower()
                if dep_key:
                    if weight > proj_deps.get(dep_key, 0.0):
                        proj_deps[dep_key] = weight
                    prev = dep_strength_global.get(dep_key)
                    if prev is None or weight > prev[0]:
                        dep_strength_global[dep_key] = (weight, canon_proj)
            elif ref.kind == "language":
                lang_key = ref.detail.strip().lower()
                if lang_key:
                    if weight > proj_langs.get(lang_key, 0.0):
                        proj_langs[lang_key] = weight
                    prev = lang_strength_global.get(lang_key)
                    if prev is None or weight > prev[0]:
                        lang_strength_global[lang_key] = (weight, canon_proj)
            elif ref.kind in ("file", "fetched_file"):
                item = _CodeEvidenceItem(
                    project_name=canon_proj,
                    kind=ref.kind,
                    detail=ref.detail,
                    path=_extract_ref_path(ref.kind, ref.detail),
                    weight=weight,
                )
                proj_code_refs.append(item)
                code_refs_global.append(item)

    return _ReportIndexes(
        project_canon=project_canon,
        project_repo=project_repo,
        ref_lookup=ref_lookup,
        ref_strength=ref_strength,
        dep_strength_by_project=dep_strength_by_project,
        dep_strength_global=dep_strength_global,
        lang_strength_by_project=lang_strength_by_project,
        lang_strength_global=lang_strength_global,
        code_refs_by_project=code_refs_by_project,
        code_refs_global=code_refs_global,
    )


def _match_builtin_in_items(
    tech: str,
    items: list[_CodeEvidenceItem],
    include_project: bool,
) -> tuple[float, str] | None:
    best_weight = 0.0
    best_label = ""
    best_rank = -1
    for item in items:
        if not _term_in_text(tech, item.detail):
            continue
        rank = 1 if item.kind == "fetched_file" else 0
        if item.weight > best_weight or (item.weight == best_weight and rank > best_rank):
            best_weight = item.weight
            best_rank = rank
            if include_project:
                best_label = f"{item.kind}: {item.path} ({item.project_name})"
            else:
                best_label = f"{item.kind}: {item.path}"
    if best_label:
        return best_weight, best_label
    return None


def _project_tech_overlap(
    project_name: str,
    startup: StartupProfile,
    indexes: _ReportIndexes,
) -> tuple[list[str], dict[str, str]]:
    proj_key = project_name.strip().lower()
    proj_deps = indexes.dep_strength_by_project.get(proj_key, {})
    proj_langs = indexes.lang_strength_by_project.get(proj_key, {})
    proj_code = indexes.code_refs_by_project.get(proj_key, [])

    overlap: list[str] = []
    sources: dict[str, str] = {}
    seen: set[str] = set()

    for tech in startup.tech_stack:
        t_low = tech.strip().lower()
        if not t_low or t_low in seen:
            continue
        if t_low in proj_deps:
            seen.add(t_low)
            overlap.append(tech)
            sources[tech] = "dependency"
        elif t_low in proj_langs:
            seen.add(t_low)
            overlap.append(tech)
            sources[tech] = "language"
        elif _is_builtin_tech(tech):
            hit = _match_builtin_in_items(tech, proj_code, include_project=False)
            if hit is not None:
                seen.add(t_low)
                overlap.append(tech)
                sources[tech] = hit[1]

    return overlap, sources


def _global_tech_overlap(
    startup: StartupProfile,
    indexes: _ReportIndexes,
) -> tuple[dict[str, float], dict[str, str]]:
    strengths: dict[str, float] = {}
    sources: dict[str, str] = {}
    lang_factor = max(0.0, settings.fit_language_overlap_weight)

    for tech in startup.tech_stack:
        t_low = tech.strip().lower()
        if not t_low or t_low in strengths:
            continue

        candidates: list[tuple[float, int, str]] = []
        if t_low in indexes.dep_strength_global:
            w, proj = indexes.dep_strength_global[t_low]
            candidates.append((w * 1.0, 3, f"dependency ({proj})"))
        if t_low in indexes.lang_strength_global:
            w, proj = indexes.lang_strength_global[t_low]
            candidates.append((w * lang_factor, 2, f"language ({proj})"))
        if _is_builtin_tech(tech):
            hit = _match_builtin_in_items(tech, indexes.code_refs_global, include_project=True)
            if hit is not None:
                rank = 1 if hit[1].startswith("fetched_file") else 0
                candidates.append((hit[0] * 1.0, rank, hit[1]))

        if candidates:
            candidates.sort(key=lambda c: (-c[0], -c[1]))
            best_w, _, best_src = candidates[0]
            strengths[t_low] = best_w
            sources[tech] = best_src

    return strengths, sources


def _build_prompt(verified_claims: list[dict], startup: StartupProfile) -> str:
    startup_payload = {
        "domain": startup.domain,
        "tech_stack": startup.tech_stack,
        "hard_problems": [p.model_dump() for p in startup.hard_problems],
    }
    return f"""Identify structured matches between the candidate's verified repository claims and the startup's domain, tech_stack, and hard_problems.

Return ONLY JSON matching this schema:
{OUTPUT_SCHEMA_DESC}

Rules:
- Do NOT compute or output any numeric scores.
- Only return grounded matches from verified_claims. If nothing in a project relates meaningfully, omit it rather than forcing weak matches.
- Use startup_problem_id equal to an exact id from startup.hard_problems ONLY when verified evidence shows relevant engineering mechanisms (never stretch generic CRUD controllers into concurrency, queue starvation, SQLite lock, or offline merge work). Leave truly unaddressed hard_problems unmatched so they surface as gaps.
- Only use startup_problem_id: null when a project's verified stack items overlap directly with startup.tech_stack (never cite unrelated libraries not in startup.tech_stack).
- Classify "relation":
  - "direct": verified work directly solves or closely mirrors that hard_problem or core domain workload.
  - "adjacent": verified work uses a neighboring engineering pattern, domain, or startup tech_stack library that applies with modest adaptation.
  - "transferable": foundational skill or architecture from a different domain that can transfer with significant adaptation.
- Every entry in "evidence_refs" MUST copy an exact (kind, detail) pair from that project_name's evidence list in verified_claims.
- Emit at most one match per (project_name, startup_problem_id) pair.
- Write one concise, factual explanation per match without inventing claims not present in verified_claims.

Input JSON:
{json.dumps({"verified_claims": verified_claims, "startup": startup_payload}, ensure_ascii=False)}"""


def _match_strength(match: FitMatch, indexes: _ReportIndexes) -> float:
    strong_w = settings.fit_strong_weight if settings.fit_strong_weight > 0 else 1.0
    proj_key = match.project_name.strip().lower()
    weights = [
        indexes.ref_strength.get((proj_key, ref.kind, ref.detail), 0.0)
        for ref in match.evidence_refs
    ]
    if not weights:
        return 0.0
    return max(0.0, min(1.0, (sum(weights) / len(weights)) / strong_w))


def _validate_matches(
    parsed: _LLMFitMatchList,
    indexes: _ReportIndexes,
    startup: StartupProfile,
    strict: bool,
) -> list[FitMatch]:
    valid_problem_ids = {p.id for p in startup.hard_problems}
    best_by_pair: dict[tuple[str, str | None], FitMatch] = {}

    for item in parsed.matches:
        proj_key = item.project_name.strip().lower()
        canon_proj = indexes.project_canon.get(proj_key)
        if canon_proj is None:
            if strict:
                raise ValueError(f"Unknown project_name in match: {item.project_name!r}")
            continue

        prob_id = item.startup_problem_id.strip() if item.startup_problem_id else None
        if prob_id is not None and prob_id not in valid_problem_ids:
            if strict:
                raise ValueError(f"Unknown startup_problem_id in match: {prob_id!r}")
            continue

        explanation = item.explanation.strip()
        if not explanation:
            if strict:
                raise ValueError(f"Empty explanation for match on {canon_proj!r}")
            continue

        if not item.evidence_refs:
            if strict:
                raise ValueError(f"Empty evidence_refs for match on {canon_proj!r}")
            continue

        validated_refs: list[EvidenceRef] = []
        seen_refs: set[tuple[str, str]] = set()
        for ref in item.evidence_refs:
            matched = indexes.ref_lookup.get((proj_key, ref.kind, ref.detail))
            if matched is None:
                if strict:
                    raise ValueError(
                        f"Match for {canon_proj!r} cites evidence_ref not found in report: ({ref.kind}, {ref.detail!r})"
                    )
                continue
            ref_id = (matched.kind, matched.detail)
            if ref_id not in seen_refs:
                seen_refs.add(ref_id)
                validated_refs.append(matched.model_copy())

        if not validated_refs:
            if strict:
                raise ValueError(f"No valid evidence_refs for match on {canon_proj!r}")
            continue

        overlap, overlap_sources = _project_tech_overlap(canon_proj, startup, indexes)
        candidate = FitMatch(
            project_name=canon_proj,
            matched_repo=indexes.project_repo.get(proj_key),
            startup_problem_id=prob_id,
            tech_overlap=overlap,
            tech_overlap_sources=overlap_sources,
            relation=item.relation,
            evidence_refs=validated_refs,
            explanation=explanation,
        )

        pair_key = (proj_key, prob_id)
        existing = best_by_pair.get(pair_key)
        if existing is None:
            best_by_pair[pair_key] = candidate
        else:
            cand_score = _relation_weight(candidate.relation) * _match_strength(candidate, indexes)
            exist_score = _relation_weight(existing.relation) * _match_strength(existing, indexes)
            if cand_score > exist_score:
                best_by_pair[pair_key] = candidate

    return list(best_by_pair.values())


def _parse_lenient(
    raw: str,
    indexes: _ReportIndexes,
    startup: StartupProfile,
) -> list[FitMatch]:
    if not raw.strip():
        return []
    try:
        data = json.loads(_extract_json_object(raw))
    except Exception:
        return []
    raw_items = data.get("matches") if isinstance(data, dict) else None
    if not isinstance(raw_items, list):
        return []

    valid_items: list[_LLMFitMatch] = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        try:
            valid_items.append(_LLMFitMatch.model_validate(entry))
        except ValidationError:
            continue
    return _validate_matches(_LLMFitMatchList(matches=valid_items), indexes, startup, strict=False)


def _compute_scores(
    matches: list[FitMatch],
    startup: StartupProfile,
    indexes: _ReportIndexes,
    global_tech_strengths: dict[str, float],
) -> tuple[float, float, float]:
    strong_w = settings.fit_strong_weight if settings.fit_strong_weight > 0 else 1.0
    problem_slots = max(len(startup.hard_problems), 1)

    if startup.hard_problems:
        problem_direct_scores = [
            max(
                (
                    _match_strength(m, indexes)
                    for m in matches
                    if m.relation == "direct" and m.startup_problem_id == p.id
                ),
                default=0.0,
            )
            for p in startup.hard_problems
        ]
        domain_fit = _clamp01(sum(problem_direct_scores) / len(startup.hard_problems))
    else:
        direct_strengths = [_match_strength(m, indexes) for m in matches if m.relation == "direct"]
        domain_fit = _clamp01(sum(direct_strengths) / max(len(matches), 1)) if matches else 0.0

    if startup.tech_stack:
        overlap_sum = sum(
            global_tech_strengths.get(t.strip().lower(), 0.0)
            for t in startup.tech_stack
        )
        overlap_score = _clamp01(overlap_sum / (len(startup.tech_stack) * strong_w))
    else:
        overlap_score = 0.0

    max_rel_w = max(
        settings.fit_relation_direct_weight,
        settings.fit_relation_adjacent_weight,
        settings.fit_relation_transferable_weight,
        1e-9,
    )
    problem_rel_scores: list[float] = []
    for p in startup.hard_problems:
        scores_for_p = [
            _relation_weight(m.relation) * _match_strength(m, indexes)
            for m in matches
            if m.startup_problem_id == p.id
        ]
        problem_rel_scores.append(max(scores_for_p) if scores_for_p else 0.0)

    tech_rel_by_proj: dict[str, float] = {}
    for m in matches:
        if m.startup_problem_id is None:
            proj_key = m.project_name.strip().lower()
            val = 0.5 * _relation_weight(m.relation) * _match_strength(m, indexes)
            if val > tech_rel_by_proj.get(proj_key, 0.0):
                tech_rel_by_proj[proj_key] = val

    relation_score = _clamp01(
        (sum(problem_rel_scores) + sum(tech_rel_by_proj.values()))
        / (problem_slots * max_rel_w)
    )

    w_ov = settings.fit_skill_overlap_weight
    w_rel = settings.fit_skill_relation_weight
    if w_ov + w_rel > 0:
        skill_fit = _clamp01((w_ov * overlap_score + w_rel * relation_score) / (w_ov + w_rel))
    else:
        skill_fit = _clamp01((overlap_score + relation_score) / 2.0)

    w_dom = settings.fit_domain_weight
    w_sk = settings.fit_skill_weight
    if w_dom + w_sk > 0:
        overall = _clamp01((w_dom * domain_fit + w_sk * skill_fit) / (w_dom + w_sk))
    else:
        overall = _clamp01((domain_fit + skill_fit) / 2.0)

    return domain_fit, skill_fit, overall


def score_fit(report: EvidenceReport, startup: StartupProfile) -> FitResult:
    verified_claims = _collect_verified_claims(report)
    indexes = _build_report_indexes(report)
    global_strengths, global_sources = _global_tech_overlap(startup, indexes)
    all_gaps = [f"{p.id}: {p.title}" for p in startup.hard_problems]

    if not verified_claims:
        return FitResult(
            startup_slug=startup.slug,
            domain_fit=0.0,
            skill_fit=0.0,
            overall=0.0,
            weak_match=True,
            matches=[],
            tech_overlap_sources={},
            gaps=all_gaps,
            llm_calls=0,
            model=settings.gemini_model,
            prompt_version=PROMPT_VERSION,
        )

    prompt = _build_prompt(verified_claims, startup)
    llm_calls = 0
    matches: list[FitMatch] = []
    first_raw = ""
    first_err: Exception | None = None

    try:
        llm_calls += 1
        first_raw = call_llm(prompt, system=SYSTEM_PROMPT, temperature=0)
        parsed = _LLMFitMatchList.model_validate_json(_strip_code_fences(first_raw))
        matches = _validate_matches(parsed, indexes, startup, strict=True)
    except Exception as err:
        invalidate_llm_cache(prompt, system=SYSTEM_PROMPT, temperature=0)
        first_err = err

    if first_err is not None:
        retry_raw = ""
        retry_messages = None
        try:
            llm_calls += 1
            if first_raw:
                retry_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": first_raw},
                    {
                        "role": "user",
                        "content": (
                            f"Validation failed with error:\n{first_err}\n\n"
                            "Fix all errors and return ONLY valid JSON matching the exact schema."
                        ),
                    },
                ]
                retry_raw = call_llm(retry_messages, system=SYSTEM_PROMPT, temperature=0)
            else:
                retry_raw = call_llm(prompt, system=SYSTEM_PROMPT, temperature=0)

            parsed_retry = _LLMFitMatchList.model_validate_json(_strip_code_fences(retry_raw))
            matches = _validate_matches(parsed_retry, indexes, startup, strict=True)
        except Exception:
            invalidate_llm_cache(
                retry_messages if retry_messages is not None else prompt,
                system=SYSTEM_PROMPT,
                temperature=0,
            )
            if retry_raw:
                matches = _parse_lenient(retry_raw, indexes, startup)
            if not matches and first_raw:
                matches = _parse_lenient(first_raw, indexes, startup)

    domain_fit, skill_fit, overall = _compute_scores(matches, startup, indexes, global_strengths)
    matched_problem_ids = {m.startup_problem_id for m in matches if m.startup_problem_id}
    gaps = [f"{p.id}: {p.title}" for p in startup.hard_problems if p.id not in matched_problem_ids]
    valid_problem_ids = {p.id for p in startup.hard_problems}
    weak_match = is_weak_match(matches, valid_problem_ids=valid_problem_ids)

    return FitResult(
        startup_slug=startup.slug,
        domain_fit=domain_fit,
        skill_fit=skill_fit,
        overall=overall,
        weak_match=weak_match,
        matches=matches,
        tech_overlap_sources=global_sources,
        gaps=gaps,
        llm_calls=llm_calls,
        model=settings.gemini_model,
        prompt_version=PROMPT_VERSION,
    )
