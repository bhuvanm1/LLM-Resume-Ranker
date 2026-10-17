"""Claude via the Anthropic API with *structured output* (forced tool use).
Every function degrades gracefully to a deterministic fallback when no API key
is set or a call fails, so the system always works offline."""
import json
from . import config, skills
from .schemas import JobProfile, ResumeProfile

_client = None


_provider_cache: str | None = None


def ollama_running() -> bool:
    try:
        import httpx
        return httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def provider() -> str:
    """Resolve once: 'anthropic' | 'ollama' | 'none'."""
    global _provider_cache
    if _provider_cache is None:
        want = config.LLM_PROVIDER
        if want == "anthropic" or (want == "auto" and config.ANTHROPIC_API_KEY):
            _provider_cache = "anthropic" if config.ANTHROPIC_API_KEY else "none"
        elif want in ("auto", "ollama") and ollama_running():
            _provider_cache = "ollama"
        else:
            _provider_cache = "none"
    return _provider_cache


def llm_available() -> bool:
    return provider() != "none"


def llm_name() -> str:
    p = provider()
    return {"anthropic": f"Claude ({config.LLM_MODEL})", "ollama": f"Ollama ({config.OLLAMA_MODEL})"}.get(p, "offline heuristics")


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _ollama_call(system: str, user: str, schema: dict) -> dict:
    """Ollama structured outputs: pass the JSON schema as `format`; the reply is constrained to it."""
    import httpx
    r = httpx.post(f"{config.OLLAMA_URL}/api/chat", timeout=300, json={
        "model": config.OLLAMA_MODEL, "stream": False, "format": schema,
        "options": {"temperature": 0},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def _structured_call(system: str, user: str, tool_name: str, schema: dict) -> dict:
    """Return a dict matching `schema`. Anthropic: forced tool use. Ollama: JSON-schema format."""
    if provider() == "ollama":
        return _ollama_call(system, user, schema)
    resp = _get_client().messages.create(
        model=config.LLM_MODEL,
        max_tokens=1500,
        temperature=0,
        system=system,
        tools=[{"name": tool_name, "description": "Return the structured result.", "input_schema": schema}],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("no structured output returned")


# ---------- prompts ----------
_SYS_EXTRACT = (
    "You are a precise recruiting data extractor. Extract ONLY what is explicitly stated. "
    "Never infer or output name, gender, age, ethnicity, nationality, marital status or photo info. "
    "Use short canonical skill names (e.g. 'Python', 'Machine Learning', 'AWS')."
)
_SYS_EXPLAIN = (
    "You are an objective hiring assistant. Explain the match using ONLY job-relevant evidence "
    "(skills, experience, achievements). Never reference or speculate about protected attributes. "
    "Be concise, specific, and actionable. Improvements must be truthful: suggest how to better "
    "present or acquire genuine skills, never to fabricate experience."
)

_SKILL_ARR = {"type": "array", "items": {"type": "string"}}


def extract_job_profile(jd: str) -> JobProfile:
    lines = skills.requirement_lines(jd)
    if llm_available():
        try:
            out = _structured_call(
                _SYS_EXTRACT,
                f"Extract the job profile from this job description:\n\n{jd}",
                "job_profile",
                {"type": "object", "properties": {
                    "title": {"type": "string"},
                    "required_skills": _SKILL_ARR,
                    "preferred_skills": _SKILL_ARR,
                    "min_years_experience": {"type": "number"}},
                 "required": ["title", "required_skills", "preferred_skills", "min_years_experience"]},
            )
            req = [skills.canonicalize(s) for s in out["required_skills"]]
            pref = [skills.canonicalize(s) for s in out["preferred_skills"] if skills.canonicalize(s) not in req]
            return JobProfile(title=out["title"], required_skills=list(dict.fromkeys(req)),
                              preferred_skills=list(dict.fromkeys(pref)),
                              min_years_experience=out["min_years_experience"], requirement_lines=lines)
        except Exception as e:  # noqa
            print(f"[llm] job extraction failed, using fallback: {e}")
    req_t, pref_t = skills.split_jd_sections(jd)
    req = skills.extract_skills(req_t)
    pref = [s for s in skills.extract_skills(pref_t) if s not in req]
    title = next((l.strip() for l in jd.splitlines() if l.strip()), "Job")[:80]
    return JobProfile(title=title, required_skills=req, preferred_skills=pref,
                      min_years_experience=skills.estimate_years(req_t) if skills._YEARS.search(req_t) else 0,
                      requirement_lines=lines)


def extract_resume_profile(redacted_text: str) -> ResumeProfile:
    if llm_available():
        try:
            out = _structured_call(
                _SYS_EXTRACT,
                f"Extract the candidate profile from this (anonymised) resume:\n\n{redacted_text}",
                "resume_profile",
                {"type": "object", "properties": {
                    "skills": _SKILL_ARR,
                    "years_experience": {"type": "number"},
                    "job_titles": _SKILL_ARR,
                    "education": _SKILL_ARR},
                 "required": ["skills", "years_experience", "job_titles", "education"]},
            )
            return ResumeProfile(
                skills=list(dict.fromkeys(skills.canonicalize(s) for s in out["skills"])),
                years_experience=out["years_experience"], job_titles=out["job_titles"][:5],
                education=out["education"][:3])
        except Exception as e:  # noqa
            print(f"[llm] resume extraction failed, using fallback: {e}")
    return ResumeProfile(skills=skills.extract_skills(redacted_text),
                         years_experience=skills.estimate_years(redacted_text),
                         job_titles=skills.extract_titles(redacted_text),
                         education=skills.extract_education(redacted_text))


def explain_and_improve(job: JobProfile, profile: ResumeProfile, matched_req: list[str],
                        missing_req: list[str], missing_pref: list[str], final_score: float,
                        evidence: list[dict]) -> tuple[str, list[str]]:
    if llm_available():
        try:
            payload = {"job_title": job.title, "score_out_of_100": round(final_score, 1),
                       "required_matched": matched_req, "required_missing": missing_req,
                       "preferred_missing": missing_pref, "candidate_years": profile.years_experience,
                       "min_years_required": job.min_years_experience, "top_evidence": evidence[:3]}
            out = _structured_call(
                _SYS_EXPLAIN,
                "Explain this ranking in 2-3 sentences and give 3-5 concrete resume improvements.\n"
                + json.dumps(payload),
                "explanation",
                {"type": "object", "properties": {
                    "explanation": {"type": "string"},
                    "improvements": {"type": "array", "items": {"type": "string"}}},
                 "required": ["explanation", "improvements"]},
            )
            return out["explanation"], out["improvements"][:5]
        except Exception as e:  # noqa
            print(f"[llm] explanation failed, using fallback: {e}")
    return _template_explain(job, profile, matched_req, missing_req, missing_pref, final_score, evidence)


def _template_explain(job, profile, matched_req, missing_req, missing_pref, score, evidence):
    total = len(matched_req) + len(missing_req)
    parts = [f"Scored {score:.0f}/100."]
    if total:
        parts.append(f"Covers {len(matched_req)}/{total} required skills"
                     + (f" ({', '.join(matched_req[:5])})." if matched_req else "."))
    if missing_req:
        parts.append(f"Missing required: {', '.join(missing_req)}.")
    if job.min_years_experience:
        gap = job.min_years_experience - profile.years_experience
        parts.append("Meets the experience requirement." if gap <= 0 else
                     f"About {gap:.0f} year(s) short of the {job.min_years_experience:.0f}-year requirement.")
    if evidence:
        parts.append(f"Strongest evidence for: \"{evidence[0]['requirement'][:90]}\".")
    tips = [f"If you genuinely have experience with {s}, add a bullet showing where and how you used it."
            for s in missing_req[:3]]
    tips += [f"Consider building or highlighting exposure to {s} (preferred)." for s in missing_pref[:1]]
    tips.append("Quantify impact (e.g. latency cut by 30%, 1M users served) in your top bullets.")
    tips.append("Mirror the job's terminology for skills you already have so they are easy to find.")
    return " ".join(parts), tips[:5]


def fairness_summary(rows: list[dict]) -> str | None:
    """Optional LLM-written plain-English summary of the audit; None -> caller uses fallback."""
    if not llm_available():
        return None
    try:
        out = _structured_call(
            "You are an AI-fairness reviewer. Summarise hiring-model audit results neutrally in 2-3 sentences.",
            json.dumps(rows), "fairness", {"type": "object", "properties": {"summary": {"type": "string"}},
                                           "required": ["summary"]})
        return out["summary"]
    except Exception:
        return None
