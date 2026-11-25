import argparse
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.llm_client import CACHE_ROOT_DIR, format_cache_summary

DEFAULT_PDF = Path.home() / "Documents" / "resume.pdf"
DEFAULT_ANSWER_PATH = ROOT / "data" / "test_answers" / "a_specific_correct.txt"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_server(base_url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"Server at {base_url} did not become healthy within {timeout}s")


def _assert_no_founder_leak(data: object, context_label: str) -> None:
    text = str(data)
    if "@example.com" in text or "x_handle" in text:
        raise AssertionError(f"Founder contact leaked in {context_label}")


def _assert_uuid4(value: str, label: str) -> None:
    parsed = uuid.UUID(value)
    if parsed.version != 4 or str(parsed) != value:
        raise AssertionError(f"Expected uuid4 string for {label}, got {value!r}")


def _print_step(method: str, path: str, resp: httpx.Response, note: str = "", label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    suffix = f" | {note}" if note else ""
    print(f"{prefix}[{resp.status_code}] {method:<6} {path}{suffix}")


def _serve_offline_server(port: int) -> None:
    import uvicorn
    from backend.app.config import settings
    from backend.app.main import app
    from backend.app.routers import candidates as candidates_router
    from backend.app.routers import sessions as sessions_router
    from backend.app.schemas import (
        AnswerScore,
        Claim,
        ClaimResult,
        DimensionScore,
        EvidenceRef,
        EvidenceReport,
        EvidencedProject,
        FitMatch,
        FitResult,
        GitHubProfile,
        ProjectItem,
        ProjectSummary,
        Question,
        QuestionSet,
        RepoEvidence,
        ResumeProfile,
        StartupProfile,
        TestSession,
        Verdict,
    )
    from backend.app.services import pipeline

    def _stub_verify(username: str) -> str:
        return username.strip().lstrip("@")

    def _stub_extract(_path: Path) -> str:
        return "Offline Fixture Resume\nProject EdgeSync\nProject QueueCraft\n"

    def _stub_parse(_text: str) -> ResumeProfile:
        return ResumeProfile(
            name="Offline Candidate",
            skills=["Python", "SQLite", "FastAPI"],
            projects=[
                ProjectItem(
                    name="EdgeSync",
                    description="Offline-first telemetry sync engine",
                    technologies=["Python", "SQLite"],
                    claims=[
                        Claim(
                            text="Implemented durable offline queue with idempotent replay",
                            type="feature",
                            source_excerpt="durable offline queue with idempotent replay",
                        )
                    ],
                    links=["https://github.com/offline/edgesync"],
                ),
                ProjectItem(
                    name="QueueCraft",
                    description="Lightweight background worker pool",
                    technologies=["Python", "FastAPI"],
                    claims=[
                        Claim(
                            text="Added exponential backoff retries for webhook delivery",
                            type="feature",
                            source_excerpt="exponential backoff retries for webhook delivery",
                        )
                    ],
                    links=["https://github.com/offline/queuecraft"],
                ),
            ],
        )

    def _stub_github(username: str, _links: list[str] | None = None) -> GitHubProfile:
        return GitHubProfile(
            username=username,
            fetched_at="2026-11-25T00:00:00Z",
            repos=[
                RepoEvidence(
                    name="edgesync",
                    full_name=f"{username}/edgesync",
                    url=f"https://github.com/{username}/edgesync",
                    languages={"Python": 12000},
                    dependencies=["fastapi", "sqlalchemy"],
                    user_commit_count=18,
                    total_commit_count=18,
                    source="resume_link",
                ),
                RepoEvidence(
                    name="queuecraft",
                    full_name=f"{username}/queuecraft",
                    url=f"https://github.com/{username}/queuecraft",
                    languages={"Python": 9000},
                    dependencies=["httpx"],
                    user_commit_count=12,
                    total_commit_count=12,
                    source="resume_link",
                ),
            ],
        )

    def _stub_report(_resume: ResumeProfile, github: GitHubProfile) -> EvidenceReport:
        return EvidenceReport(
            results=[
                ClaimResult(
                    claim_text="Implemented durable offline queue with idempotent replay",
                    claim_type="feature",
                    source_section="projects",
                    project_name="EdgeSync",
                    matched_repo=f"{github.username}/edgesync",
                    verdict=Verdict.STRONG,
                    evidence=[EvidenceRef(kind="file", detail="edgesync/outbox.py:42", independent=True)],
                    checked=["edgesync/outbox.py"],
                    reason="Verified durable SQLite outbox replay logic",
                    model="offline-stub",
                ),
                ClaimResult(
                    claim_text="Added exponential backoff retries for webhook delivery",
                    claim_type="feature",
                    source_section="projects",
                    project_name="QueueCraft",
                    matched_repo=f"{github.username}/queuecraft",
                    verdict=Verdict.STRONG,
                    evidence=[EvidenceRef(kind="file", detail="queuecraft/retry.py:18", independent=True)],
                    checked=["queuecraft/retry.py"],
                    reason="Verified retry loop with jitter",
                    model="offline-stub",
                ),
                ClaimResult(
                    claim_text="Reduced tail p99 sync latency by 40%",
                    claim_type="numeric_metric",
                    source_section="projects",
                    project_name="EdgeSync",
                    matched_repo=f"{github.username}/edgesync",
                    verdict=Verdict.INSUFFICIENT_EVIDENCE,
                    evidence=[],
                    checked=["README.md"],
                    reason="Claim numbers not found verbatim in README or benchmark files",
                    model="offline-stub",
                ),
                ClaimResult(
                    claim_text="Built custom hardware telemetry aggregator",
                    claim_type="feature",
                    source_section="projects",
                    project_name="LegacyTool",
                    matched_repo=None,
                    verdict=Verdict.INSUFFICIENT_EVIDENCE,
                    evidence=[],
                    checked=[],
                    reason="no linked public repo",
                    candidate_action="link a public repo or explain in a dispute",
                    model="offline-stub",
                ),
            ],
            projects=[
                ProjectSummary(
                    project_name="EdgeSync",
                    repo=f"{github.username}/edgesync",
                    counts={"STRONG": 1, "MODERATE": 0, "INSUFFICIENT_EVIDENCE": 1, "UNSUPPORTED": 0, "CONTRADICTED": 0},
                    strong_feature_claims=1,
                    strong_independent_features=1,
                    evidenced=True,
                    evidenced_reason="1 STRONG independent feature claim(s)",
                ),
                ProjectSummary(
                    project_name="QueueCraft",
                    repo=f"{github.username}/queuecraft",
                    counts={"STRONG": 1, "MODERATE": 0, "INSUFFICIENT_EVIDENCE": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0},
                    strong_feature_claims=1,
                    strong_independent_features=1,
                    evidenced=True,
                    evidenced_reason="1 STRONG independent feature claim(s)",
                ),
                ProjectSummary(
                    project_name="LegacyTool",
                    repo=None,
                    counts={"STRONG": 0, "MODERATE": 0, "INSUFFICIENT_EVIDENCE": 1, "UNSUPPORTED": 0, "CONTRADICTED": 0},
                    evidenced=False,
                    evidenced_reason="0 STRONG independent feature claims and 0/3 MODERATE feature claims",
                ),
            ],
            evidenced_projects=[
                EvidencedProject(project_name="EdgeSync", repo=f"{github.username}/edgesync", reason="1 STRONG independent feature claim(s)"),
                EvidencedProject(project_name="QueueCraft", repo=f"{github.username}/queuecraft", reason="1 STRONG independent feature claim(s)"),
            ],
            evidence_gate_passed=True,
            model="offline-stub",
            prompt_version="offline-v1",
            created_at="2026-11-25T00:00:00Z",
        )

    def _stub_fit(_report: EvidenceReport, startup: StartupProfile) -> FitResult:
        if startup.slug == "verifield":
            prob_id = startup.hard_problems[0].id if startup.hard_problems else "offline-sync"
            return FitResult(
                startup_slug=startup.slug,
                domain_fit=0.82,
                skill_fit=0.79,
                overall=0.8065,
                weak_match=False,
                matches=[
                    FitMatch(
                        project_name="EdgeSync",
                        startup_problem_id=prob_id,
                        tech_overlap=["Python", "SQLite"],
                        tech_overlap_sources={"Python": "language", "SQLite": "dependency"},
                        relation="direct",
                        evidence_refs=[EvidenceRef(kind="file", detail="edgesync/outbox.py:42")],
                        explanation="Direct offline synchronization and SQLite durability match",
                    )
                ],
                gaps=[],
                model="offline-stub",
                prompt_version="offline-v1",
            )
        return FitResult(
            startup_slug=startup.slug,
            domain_fit=0.1,
            skill_fit=0.05,
            overall=0.0775,
            weak_match=True,
            matches=[],
            gaps=["Limited domain overlap in offline stub"],
            model="offline-stub",
            prompt_version="offline-v1",
        )

    def _stub_questions(fit: FitResult, startup: StartupProfile, **_kwargs) -> QuestionSet:
        if fit.weak_match or startup.slug == "voltcore":
            return QuestionSet(
                startup_slug=startup.slug,
                questions=[],
                weak_match=True,
                reason="No direct or adjacent verified project overlap",
                closest_verified_work=["EdgeSync"],
                model="offline-stub",
                verifier_model="offline-stub",
                prompt_version="offline-v1",
            )
        prob_id = startup.hard_problems[0].id if startup.hard_problems else "offline-sync"
        return QuestionSet(
            startup_slug=startup.slug,
            questions=[
                Question(
                    id="q1",
                    text="In EdgeSync, how do you prevent duplicate mutations when a client reconnects after a partial batch timeout?",
                    project_name="EdgeSync",
                    grounded_in=[EvidenceRef(kind="file", detail="edgesync/outbox.py:42")],
                    startup_problem_id=prob_id,
                    context_chars=420,
                    verification_verdict="YES",
                    verification_reason="Grounded in outbox sequence handling",
                ),
                Question(
                    id="q2",
                    text="In QueueCraft, what failure modes drove your retry backoff ceiling and jitter choices?",
                    project_name="QueueCraft",
                    grounded_in=[EvidenceRef(kind="file", detail="queuecraft/retry.py:18")],
                    startup_problem_id=prob_id,
                    context_chars=380,
                    verification_verdict="YES",
                    verification_reason="Grounded in retry loop design",
                ),
            ],
            weak_match=False,
            candidates_generated=4,
            questions_verified=2,
            model="offline-stub",
            verifier_model="offline-stub",
            prompt_version="offline-v1",
        )

    def _stub_followup(question: Question, _answer: str) -> str:
        return f"How would your design in {question.project_name} handle storage corruption mid-sync?"

    def _stub_evaluate(
        startup_slug: str,
        questions: list[Question],
        answers: dict[str, str],
        followup_answers: dict[str, str],
        followups: dict[str, str],
        seconds_per_question: dict[str, float],
        used_dispute: bool = False,
    ) -> TestSession:
        threshold = settings.min_test_score_with_dispute if used_dispute else settings.min_test_score
        scores = [
            AnswerScore(
                question_id=q.id,
                dimensions=[
                    DimensionScore(dimension="specificity_to_own_design", score=3, reason="Specific implementation detail", quotes=["idempotent"]),
                    DimensionScore(dimension="tradeoff_reasoning", score=3, reason="Clear durability vs latency tradeoff", quotes=["retry"]),
                    DimensionScore(dimension="technical_correctness", score=3, reason="Sound mechanism", quotes=["SQLite"]),
                    DimensionScore(dimension="problem_understanding", score=2, reason="Solid edge-case handling", quotes=["timeout"]),
                ],
                score=0.90,
                followup_score=0.86 if q.id in followup_answers else 0.0,
                combined_score=0.888 if q.id in followup_answers else 0.90,
            )
            for q in questions
        ]
        answered = [s.combined_score or s.score for s in scores if s.question_id in answers]
        total = round(sum(answered) / len(scores), 4) if scores and answered else 0.0
        return TestSession(
            startup_slug=startup_slug,
            questions=questions,
            answers=answers,
            followups=followups,
            followup_answers=followup_answers,
            scores=scores,
            total=total,
            passed=total >= threshold and len(answered) == len(questions),
            used_dispute=used_dispute,
            seconds_per_question=seconds_per_question,
            model="offline-stub",
            prompt_version="offline-v1",
        )

    def _offline_run_analysis(candidate_id: str, pdf_bytes: bytes) -> None:
        pipeline.run_analysis(candidate_id, pdf_bytes, use_stage_cache=False)

    candidates_router.verify_github_username = _stub_verify
    candidates_router.run_analysis = _offline_run_analysis
    pipeline.extract_text = _stub_extract
    pipeline.parse_resume = _stub_parse
    pipeline.analyze_github = _stub_github
    pipeline.check_resume = _stub_report
    pipeline.score_fit = _stub_fit
    sessions_router.generate_questions = _stub_questions
    sessions_router.generate_followup = _stub_followup
    sessions_router.evaluate_session = _stub_evaluate

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--username", default="Sahith69")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--startup", default="verifield")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--_serve-offline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._serve_offline:
        _serve_offline_server(args.port)
        return

    if args.offline and args.base_url:
        parser.error("--offline cannot be combined with --base-url")

    label = "OFFLINE" if args.offline else ""
    if args.offline:
        print("OFFLINE mode: LLM layers stubbed with fixtures; exercising real HTTP API and SQLite DB.")

    pdf_path = Path(args.pdf).expanduser().resolve()
    if pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()
        pdf_name = pdf_path.name
    elif args.offline:
        pdf_bytes = b"%PDF-1.4\n%offline fixture resume\n"
        pdf_name = "offline_resume.pdf"
    else:
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")

    server_proc = None
    base_url = args.base_url.rstrip("/") if args.base_url else None
    if base_url is None:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        if args.offline:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_serve-offline",
                "--port",
                str(port),
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
        server_proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _wait_for_server(base_url)

    candidate_id = None
    try:
        with httpx.Client(base_url=base_url, timeout=60.0) as client:
            r = client.get("/health")
            _print_step("GET", "/health", r, label=label)

            r = client.post(
                "/candidates",
                json={
                    "github_username": args.username,
                    "processing_notice_accepted": True,
                    "data_consent": False,
                },
            )
            _print_step("POST", "/candidates", r, label=label)
            r.raise_for_status()
            candidate_id = r.json()["id"]
            _assert_uuid4(candidate_id, "candidate_id")

            r_patch_on = client.patch(
                f"/candidates/{candidate_id}/consent",
                json={"data_consent": True},
            )
            _print_step("PATCH", f"/candidates/{candidate_id}/consent (opt-in)", r_patch_on, label=label)
            r_patch_on.raise_for_status()

            r_patch_off = client.patch(
                f"/candidates/{candidate_id}/consent",
                json={"data_consent": False},
            )
            _print_step("PATCH", f"/candidates/{candidate_id}/consent (withdraw)", r_patch_off, label=label)
            r_patch_off.raise_for_status()

            r = client.post(
                f"/candidates/{candidate_id}/analyze",
                files={"file": (pdf_name, pdf_bytes, "application/pdf")},
            )
            _print_step("POST", f"/candidates/{candidate_id}/analyze", r, label=label)
            r.raise_for_status()

            analysis_data = None
            for _ in range(600):
                r = client.get(f"/candidates/{candidate_id}/analysis")
                analysis_data = r.json()
                status = analysis_data.get("status")
                if status in ("done", "failed"):
                    _print_step(
                        "GET",
                        f"/candidates/{candidate_id}/analysis",
                        r,
                        f"status={status}, gate={analysis_data.get('evidence_gate_passed')}, prioritized={len(analysis_data.get('rejection_feedback', []))}, total_gaps={len(analysis_data.get('all_gaps', []))}",
                        label=label,
                    )
                    break
                time.sleep(0.5)
            else:
                raise TimeoutError("Analysis polling timed out")

            _assert_no_founder_leak(analysis_data, "GET /analysis")
            if len(analysis_data.get("rejection_feedback", [])) > 5:
                raise AssertionError("rejection_feedback exceeded 5 prioritized actions")
            if analysis_data["status"] != "done":
                raise RuntimeError(
                    f"Analysis failed at {analysis_data.get('failing_stage')}: {analysis_data.get('error')}"
                )

            r = client.get(f"/candidates/{candidate_id}/startups")
            _print_step(
                "GET",
                f"/candidates/{candidate_id}/startups",
                r,
                f"count={len(r.json().get('startups', []))}",
                label=label,
            )
            r.raise_for_status()
            _assert_no_founder_leak(r.json(), "GET /startups")

            r = client.post(
                "/sessions",
                json={"candidate_id": candidate_id, "startup_slug": "voltcore"},
            )
            _print_step(
                "POST",
                "/sessions (voltcore)",
                r,
                f"weak_match={r.json().get('weak_match') if r.status_code == 201 else r.text}",
                label=label,
            )
            if r.status_code in (200, 201):
                _assert_uuid4(r.json()["session_id"], "session_id (voltcore)")

            r = client.post(
                "/sessions",
                json={"candidate_id": candidate_id, "startup_slug": args.startup},
            )
            _print_step("POST", f"/sessions ({args.startup})", r, label=label)

            if r.status_code in (200, 201):
                sess = r.json()
                _assert_no_founder_leak(sess, "POST /sessions")
                sess_id = sess["session_id"]
                _assert_uuid4(sess_id, f"session_id ({args.startup})")
                questions = sess.get("questions", [])
                if questions:
                    answer_text = (
                        DEFAULT_ANSWER_PATH.read_text(encoding="utf-8").strip()
                        if DEFAULT_ANSWER_PATH.exists()
                        else "Implemented an idempotent SQLite outbox queue with bounded exponential backoff retries."
                    )
                    for q in questions:
                        r_ans = client.post(
                            f"/sessions/{sess_id}/answer",
                            json={"question_id": q["id"], "answer": answer_text},
                        )
                        _print_step("POST", f"/sessions/{sess_id}/answer ({q['id']})", r_ans, label=label)
                        r_ans.raise_for_status()

                        r_fu = client.post(
                            f"/sessions/{sess_id}/followup",
                            json={"question_id": q["id"], "answer": answer_text},
                        )
                        _print_step("POST", f"/sessions/{sess_id}/followup ({q['id']})", r_fu, label=label)
                        r_fu.raise_for_status()

                    r_res = client.get(f"/sessions/{sess_id}/result")
                    _print_step(
                        "GET",
                        f"/sessions/{sess_id}/result",
                        r_res,
                        f"total={r_res.json().get('total')}, passed={r_res.json().get('passed')}",
                        label=label,
                    )
                    _assert_no_founder_leak(r_res.json(), "GET /result")

                    r_unl = client.post(f"/sessions/{sess_id}/unlock")
                    _print_step(
                        "POST",
                        f"/sessions/{sess_id}/unlock",
                        r_unl,
                        f"unlocked={r_unl.json().get('unlocked') if r_unl.status_code == 200 else r_unl.text}",
                        label=label,
                    )
                    if args.offline and r_unl.status_code != 200:
                        raise AssertionError(f"Expected 200 on OFFLINE unlock, got {r_unl.status_code}: {r_unl.text}")

            r_del = client.delete(f"/candidates/{candidate_id}")
            _print_step("DELETE", f"/candidates/{candidate_id}", r_del, label=label)
            r_del.raise_for_status()

            r_gone = client.get(f"/candidates/{candidate_id}/analysis")
            _print_step("GET", f"/candidates/{candidate_id}/analysis (after delete)", r_gone, label=label)
            if r_gone.status_code != 404:
                raise AssertionError(f"Expected 404 after delete, got {r_gone.status_code}")
            candidate_id = None
    finally:
        if candidate_id is not None:
            try:
                httpx.delete(f"{base_url}/candidates/{candidate_id}", timeout=5.0)
            except Exception:
                pass
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    summary = format_cache_summary(CACHE_ROOT_DIR)
    if args.offline:
        print(f"OFFLINE smoke run complete | {summary}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
