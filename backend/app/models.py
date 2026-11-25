from datetime import datetime, timezone
import uuid
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship
from backend.app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(String(36), primary_key=True, index=True, default=new_uuid)
    github_username = Column(String, index=True, nullable=False)
    processing_notice_accepted = Column(Boolean, nullable=False)
    processing_notice_accepted_at = Column(DateTime, nullable=True)
    data_consent = Column(Boolean, nullable=False)
    consent_at = Column(DateTime, nullable=True)
    consent_withdrawn_at = Column(DateTime, nullable=True)
    analysis_status = Column(String, nullable=True)
    failing_stage = Column(String, nullable=True)
    analysis_error = Column(Text, nullable=True)
    analysis_updated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    resumes = relationship("ResumeProfileRecord", back_populates="candidate", cascade="all, delete-orphan")
    github_profiles = relationship("GitHubProfileRecord", back_populates="candidate", cascade="all, delete-orphan")
    evidence_reports = relationship("EvidenceReportRecord", back_populates="candidate", cascade="all, delete-orphan")
    fit_results = relationship("FitResultRecord", back_populates="candidate", cascade="all, delete-orphan")
    test_sessions = relationship("TestSessionRecord", back_populates="candidate", cascade="all, delete-orphan")
    unlocks = relationship("UnlockRecord", back_populates="candidate", cascade="all, delete-orphan")


class ResumeProfileRecord(Base):
    __tablename__ = "resume_profiles"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    raw_text = Column(Text, nullable=False)
    parsed_profile = Column(JSON, nullable=False)
    model = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="resumes")


class GitHubProfileRecord(Base):
    __tablename__ = "github_profiles"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    username = Column(String, nullable=False)
    profile_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="github_profiles")


class EvidenceReportRecord(Base):
    __tablename__ = "evidence_reports"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    report_data = Column(JSON, nullable=False)
    evidence_gate_passed = Column(Boolean, nullable=False, default=False)
    model = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="evidence_reports")


class FitResultRecord(Base):
    __tablename__ = "fit_results"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    startup_slug = Column(String, nullable=False, index=True)
    domain_fit = Column(Float, nullable=False)
    skill_fit = Column(Float, nullable=False)
    overall = Column(Float, nullable=False)
    weak_match = Column(Boolean, nullable=False, default=False)
    fit_data = Column(JSON, nullable=False)
    model = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="fit_results")


class TestSessionRecord(Base):
    __tablename__ = "test_sessions"

    id = Column(String(36), primary_key=True, index=True, default=new_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    startup_slug = Column(String, nullable=False, index=True)
    weak_match = Column(Boolean, nullable=False, default=False)
    weak_match_reason = Column(Text, nullable=True)
    closest_verified_work = Column(JSON, nullable=False, default=list)
    questions = Column(JSON, nullable=False, default=list)
    answers = Column(JSON, nullable=False, default=dict)
    followups = Column(JSON, nullable=False, default=dict)
    followup_answers = Column(JSON, nullable=False, default=dict)
    scores = Column(JSON, nullable=False, default=list)
    total = Column(Float, nullable=True)
    passed = Column(Boolean, nullable=False, default=False)
    used_dispute = Column(Boolean, nullable=False, default=False)
    seconds_per_question = Column(JSON, nullable=False, default=dict)
    question_started_at = Column(JSON, nullable=False, default=dict)
    model = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="test_sessions")


class UnlockRecord(Base):
    __tablename__ = "unlocks"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    startup_slug = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="unlocks")
