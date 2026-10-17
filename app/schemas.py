"""Pydantic models = the structured-output contract for the API and the LLM."""
from typing import Optional
from pydantic import BaseModel, Field


class JobProfile(BaseModel):
    title: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: float = 0
    requirement_lines: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    skills: list[str] = Field(default_factory=list)
    years_experience: float = 0
    job_titles: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    semantic: float          # 0-1, embedding similarity
    skills: float            # 0-1, weighted required/preferred coverage
    experience: float        # 0-1, fit vs. required years
    final: float             # 0-100


class Evidence(BaseModel):
    requirement: str
    best_resume_snippet: str
    similarity: float


class BiasAudit(BaseModel):
    flagged_terms: list[str]            # protected-attribute signals found & redacted
    score_with_pii: float               # 0-100, scored on raw text
    score_blind: float                  # 0-100, scored on redacted text (used for ranking)
    score_delta: float                  # blind - raw
    counterfactual_gender_delta: float  # score change when gendered terms are swapped
    rank_with_pii: int
    rank_blind: int
    verdict: str


class CandidateResult(BaseModel):
    rank: int
    candidate: str
    scores: ScoreBreakdown
    profile: ResumeProfile
    matched_required: list[str]
    matched_preferred: list[str]
    missing_required: list[str]
    missing_preferred: list[str]
    evidence: list[Evidence]
    explanation: str
    improvements: list[str]
    bias: BiasAudit


class RankResponse(BaseModel):
    job: JobProfile
    llm_used: bool
    llm_name: str = "offline heuristics"
    embedding_backend: str
    weights: dict[str, float]
    results: list[CandidateResult]
    fairness_summary: str


class ResumeIn(BaseModel):
    name: str
    text: str


class RankRequest(BaseModel):
    job_description: str
    resumes: list[ResumeIn]
    top_k: Optional[int] = None
