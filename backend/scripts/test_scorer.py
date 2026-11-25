import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.schemas import AnswerScore, Question, QuestionSet
from backend.app.services.answer_scorer import generate_followup, score_answer
from backend.app.services.llm_client import (
    CACHE_ROOT_DIR,
    format_cache_summary,
    get_llm_cache_stats,
    reset_llm_cache_stats,
    set_llm_cache_enabled,
)

DEFAULT_QUESTION = ROOT / "data" / "test_questions" / "verifield_ood.json"
DEFAULT_ANSWERS_DIR = ROOT / "data" / "test_answers"


def _load_question(path: Path) -> tuple[Question, str]:
    raw = path.expanduser().resolve().read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, dict) and "questions" in data:
        qset = QuestionSet.model_validate(data)
        if not qset.questions:
            raise ValueError(f"QuestionSet in {path} has no questions")
        q = qset.questions[0]
    else:
        q = Question.model_validate(data)
    ctx = (
        (data.get("context") if isinstance(data, dict) else None)
        or q.context_preview
        or "; ".join(ref.detail for ref in q.grounded_in)
    )
    return q, ctx


def _load_answer_files(targets: list[str]) -> list[Path]:
    if not targets:
        files = sorted(DEFAULT_ANSWERS_DIR.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files in {DEFAULT_ANSWERS_DIR}")
        return files
    out: list[Path] = []
    for raw in targets:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            out.extend(sorted(p.glob("*.txt")))
        else:
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question_json", nargs="?", default=str(DEFAULT_QUESTION))
    parser.add_argument("answers", nargs="*")
    parser.add_argument("--no-llm-cache", action="store_true")
    parser.add_argument("--with-followup", action="store_true")
    args = parser.parse_args()

    reset_llm_cache_stats()
    set_llm_cache_enabled(not args.no_llm_cache)

    question, context = _load_question(Path(args.question_json))
    answer_files = _load_answer_files(args.answers)

    print(f"Question ({question.id}): {question.project_name} -> {question.startup_problem_id}")
    print(f"Scorer model: {settings.SCORER_MODEL} | min_test_score={settings.min_test_score:.2f} | with_dispute={settings.min_test_score_with_dispute:.2f}")

    results: list[tuple[str, AnswerScore, str | None]] = []
    for path in answer_files:
        ans_text = path.read_text(encoding="utf-8")
        scored = score_answer(question, context, ans_text)
        followup_q = None
        if args.with_followup and scored.rejected_reason is None:
            followup_q = generate_followup(question, ans_text)
        results.append((path.name, scored, followup_q))

        print("=" * 80)
        print(f"Answer: {path.name} ({len(ans_text.strip())} chars) | Score: {scored.score:.4f}")
        if scored.rejected_reason:
            print(f"Rejected: {scored.rejected_reason}")
        if scored.generic_flags:
            print(f"Generic Flags: {scored.generic_flags}")
        else:
            print("Generic Flags: []")
        for dim in scored.dimensions:
            print(f"  - {dim.dimension}: {dim.score}/3 | {dim.reason}")
            for qt in dim.quotes:
                print(f"      quote: {qt!r}")
        if followup_q:
            print(f"Follow-up: {followup_q}")

    stats = get_llm_cache_stats()
    total_llm_calls = stats["hits"] + stats["misses"]
    cache_hits = stats["hits"]

    print("=" * 80)
    print(f"{'Answer File':<30} {'Score':>7} {'Spec':>5} {'Trade':>6} {'Tech':>5} {'Prob':>5} {'Flags':>6} {'Pass?':<6}")
    print("-" * 80)
    for name, sc, _ in results:
        dim_map = {d.dimension: d.score for d in sc.dimensions}
        pass_str = "PASS" if sc.score >= settings.min_test_score else "FAIL"
        print(
            f"{name:<30} {sc.score:>7.4f} "
            f"{dim_map.get('specificity_to_own_design', 0):>5} "
            f"{dim_map.get('tradeoff_reasoning', 0):>6} "
            f"{dim_map.get('technical_correctness', 0):>5} "
            f"{dim_map.get('problem_understanding', 0):>5} "
            f"{len(sc.generic_flags):>6} {pass_str:<6}"
        )
    print("=" * 80)
    print(f"llm_calls={total_llm_calls}, cache_hits={cache_hits}, api_misses={stats['misses']}")
    print(format_cache_summary(CACHE_ROOT_DIR))


if __name__ == "__main__":
    main()
