from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator

ClaimType = Literal[
    "technology",
    "numeric_metric",
    "feature",
    "deployment",
    "timeline",
    "other",
]


class Claim(BaseModel):
    text: str
    type: ClaimType
    source_excerpt: str
    ambiguous: bool = False


class EducationItem(BaseModel):
    institution: str
    degree: str | None = None
    year: str | None = None


class ExperienceItem(BaseModel):
    company: str
    role: str
    duration: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    name: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)


class UnlinkedIdentity(BaseModel):
    name: str
    email: str | None = None


class RepoEvidence(BaseModel):
    name: str
    full_name: str
    url: str
    description: str | None = None
    homepage: str | None = None
    is_fork: bool = False
    parent_full_name: str | None = None
    commits_ahead_of_parent: int | None = None
    languages: dict[str, int] = Field(default_factory=dict)
    topics: list[str] = Field(default_factory=list)
    stars: int = 0
    readme_excerpt: str | None = None
    file_tree: list[str] = Field(default_factory=list)
    file_tree_truncated: bool = False
    total_file_count: int = 0
    dependencies: list[str] = Field(default_factory=list)
    user_commit_count: int = 0
    unlinked_commit_count: int = 0
    unlinked_identities: list[UnlinkedIdentity] = Field(default_factory=list)
    other_author_count: int = 0
    confirmed_email_commit_count: int = 0
    total_commit_count: int = 0
    first_commit_at: str | None = None
    last_commit_at: str | None = None
    active_days: int = 0
    largest_commit_share: float | None = None
    raw_largest_commit_share: float | None = None
    merged_prs_to_parent: int | None = None
    source: Literal["resume_link", "recent_list"]
    fetch_error: str | None = None


class GitHubProfile(BaseModel):
    username: str
    repos: list[RepoEvidence] = Field(default_factory=list)
    fetched_at: str


class Verdict(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


EvidenceKind = Literal[
    "dependency",
    "language",
    "file",
    "readme",
    "commit_stat",
    "homepage",
    "fetched_file",
]


class EvidenceRef(BaseModel):
    kind: EvidenceKind
    detail: str
    independent: bool = False


class ClaimResult(BaseModel):
    claim_text: str
    claim_type: ClaimType
    source_section: Literal["experience", "projects"]
    project_name: str
    matched_repo: str | None = None
    suggested_repo: str | None = None
    verdict: Verdict
    evidence: list[EvidenceRef] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)
    reason: str
    signals: list[str] = Field(default_factory=list)
    candidate_action: str | None = None
    ref_relevance: str | None = None
    verification_answer: str | None = None
    model: str = ""


class ProjectSummary(BaseModel):
    project_name: str
    repo: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    strong_feature_claims: int = 0
    strong_independent_features: int = 0
    moderate_feature_claims: int = 0
    evidenced: bool = False
    evidenced_reason: str = ""


class EvidencedProject(BaseModel):
    project_name: str
    repo: str | None = None
    reason: str
    qualifying_claims: list[str] = Field(default_factory=list)
    qualifying_refs: list[EvidenceRef] = Field(default_factory=list)


class EvidenceReport(BaseModel):
    results: list[ClaimResult] = Field(default_factory=list)
    projects: list[ProjectSummary] = Field(default_factory=list)
    evidenced_projects: list[EvidencedProject] = Field(default_factory=list)
    evidence_gate_passed: bool = False
    llm_calls: int = 0
    fetch_file_calls: int = 0
    model: str
    prompt_version: str
    created_at: str


class HardProblem(BaseModel):
    id: str
    title: str
    description: str


class FounderContact(BaseModel):
    name: str
    role: str
    email: str
    x_handle: str | None = None
    linkedin: str | None = None


class StartupProfile(BaseModel):
    slug: str
    name: str
    one_liner: str
    domain: str
    tech_stack: list[str] = Field(default_factory=list)
    hard_problems: list[HardProblem] = Field(default_factory=list)
    founder: FounderContact


class Question(BaseModel):
    id: str
    text: str
    project_name: str
    grounded_in: list[EvidenceRef] = Field(default_factory=list)
    startup_problem_id: str
    time_limit_minutes: int = 15
    context_chars: int = 0
    context_preview: str = ""
    verification_verdict: Literal["YES", "PARTIAL"] | None = None
    verification_reason: str | None = None


class DroppedQuestion(BaseModel):
    id: str
    text: str
    project_name: str
    startup_problem_id: str
    cited_files: list[str] = Field(default_factory=list)
    verdict: str
    reason: str


class QuestionSet(BaseModel):
    startup_slug: str
    questions: list[Question] = Field(default_factory=list)
    weak_match: bool = False
    reason: str | None = None
    closest_verified_work: list[str] = Field(default_factory=list)
    candidates_generated: int = 0
    questions_verified: int = 0
    questions_dropped: int = 0
    dropped_questions: list[DroppedQuestion] = Field(default_factory=list)
    dropped_verdict_reasons: list[str] = Field(default_factory=list)
    fetch_count: int = 0
    cached_fetches: int = 0
    llm_calls: int = 0
    model: str
    verifier_model: str | None = None
    prompt_version: str


RelationType = Literal["direct", "adjacent", "transferable"]


class FitMatch(BaseModel):
    project_name: str
    matched_repo: str | None = None
    startup_problem_id: str | None = None
    tech_overlap: list[str] = Field(default_factory=list)
    tech_overlap_sources: dict[str, str] = Field(default_factory=dict)
    relation: RelationType
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    explanation: str


class FitResult(BaseModel):
    startup_slug: str
    domain_fit: float = Field(ge=0.0, le=1.0)
    skill_fit: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    weak_match: bool = False
    matches: list[FitMatch] = Field(default_factory=list)
    tech_overlap_sources: dict[str, str] = Field(default_factory=dict)
    gaps: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    model: str
    prompt_version: str


DimensionName = Literal[
    "specificity_to_own_design",
    "tradeoff_reasoning",
    "technical_correctness",
    "problem_understanding",
]


class DimensionScore(BaseModel):
    dimension: DimensionName
    score: int = Field(ge=0, le=3)
    reason: str
    quotes: list[str] = Field(default_factory=list, max_length=2)


class AnswerScore(BaseModel):
    question_id: str
    dimensions: list[DimensionScore] = Field(default_factory=list)
    generic_flags: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)
    rejected_reason: str | None = None
    followup_score: float | None = Field(default=None, ge=0.0, le=1.0)
    combined_score: float | None = Field(default=None, ge=0.0, le=1.0)


class TestSession(BaseModel):
    startup_slug: str
    questions: list[Question] = Field(default_factory=list)
    answers: dict[str, str] = Field(default_factory=dict)
    followups: dict[str, str] = Field(default_factory=dict)
    followup_answers: dict[str, str] = Field(default_factory=dict)
    scores: list[AnswerScore] = Field(default_factory=list)
    total: float = Field(ge=0.0, le=1.0)
    passed: bool
    used_dispute: bool = False
    seconds_per_question: dict[str, float] = Field(default_factory=dict)
    llm_calls: int = 0
    model: str
    prompt_version: str


class CandidateCreate(BaseModel):
    github_username: str
    processing_notice_accepted: bool
    data_consent: bool

    @field_validator("github_username")
    @classmethod
    def _clean_username(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("github_username must not be empty")
        return clean

    @field_validator("processing_notice_accepted", "data_consent", mode="before")
    @classmethod
    def _strict_bool_consent(cls, v: object) -> bool:
        if not isinstance(v, bool):
            raise ValueError("Flag must be an explicit boolean true or false")
        return v


class ConsentUpdate(BaseModel):
    data_consent: bool
    processing_notice_accepted: bool | None = None

    @field_validator("data_consent", mode="before")
    @classmethod
    def _strict_bool_data_consent(cls, v: object) -> bool:
        if not isinstance(v, bool):
            raise ValueError("data_consent must be an explicit boolean true or false")
        return v

    @field_validator("processing_notice_accepted", mode="before")
    @classmethod
    def _strict_bool_notice(cls, v: object) -> bool | None:
        if v is not None and not isinstance(v, bool):
            raise ValueError("processing_notice_accepted must be an explicit boolean true or false")
        return v


class SessionCreate(BaseModel):
    candidate_id: str
    startup_slug: str
    used_dispute: bool = False

    @field_validator("candidate_id")
    @classmethod
    def _clean_candidate_id(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("candidate_id must not be empty")
        return clean


class AnswerSubmit(BaseModel):
    question_id: str
    answer: str
