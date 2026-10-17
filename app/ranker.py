"""Ranking pipeline: extract -> embed -> semantic search -> score -> explain -> audit."""
import numpy as np
from . import config, skills, llm, bias
from .embeddings import embed, cosine_matrix
from .schemas import (BiasAudit, CandidateResult, Evidence, JobProfile, RankResponse,
                      ResumeProfile, ScoreBreakdown)

# cosine ranges differ per embedding model; rescale to a 0-1 "relevance" feel
_RESCALE = {"tfidf": (0.0, 0.40), "sentence-transformers": (0.15, 0.70),
            "ollama": (0.35, 0.80)}


def _rescale(x: float, backend: str) -> float:
    lo, hi = next(v for k, v in _RESCALE.items() if backend.startswith(k))
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def skill_score(job: JobProfile, have: list[str]):
    have_l = {s.lower() for s in have}
    mr = [s for s in job.required_skills if s.lower() in have_l]
    xr = [s for s in job.required_skills if s.lower() not in have_l]
    mp = [s for s in job.preferred_skills if s.lower() in have_l]
    xp = [s for s in job.preferred_skills if s.lower() not in have_l]
    req_cov = len(mr) / len(job.required_skills) if job.required_skills else None
    pref_cov = len(mp) / len(job.preferred_skills) if job.preferred_skills else None
    if req_cov is not None and pref_cov is not None:
        score = 0.8 * req_cov + 0.2 * pref_cov
    else:
        score = req_cov if req_cov is not None else (pref_cov if pref_cov is not None else 0.5)
    return score, mr, xr, mp, xp


def experience_score(job: JobProfile, years: float) -> float:
    if job.min_years_experience <= 0:
        return 1.0 if years > 0 else 0.5
    return float(min(years / job.min_years_experience, 1.0))


def combine(sem: float, sk: float, exp: float) -> float:
    w = config.W_SEMANTIC + config.W_SKILLS + config.W_EXPERIENCE
    return 100 * (config.W_SEMANTIC * sem + config.W_SKILLS * sk + config.W_EXPERIENCE * exp) / w


def rank_candidates(job_description: str, resumes: list[tuple[str, str]], top_k: int | None = None) -> RankResponse:
    job = llm.extract_job_profile(job_description)

    # ---- build the text variants: blind (used for ranking), raw, gender-swapped raw (audit) ----
    cands = []
    for name, raw in resumes:
        blind, flagged = bias.redact(raw, name)
        cands.append({"name": name, "raw": raw, "blind": blind, "swap": bias.gender_swap(raw),
                      "flagged": flagged})

    # ---- embed everything in ONE batch ----
    texts: list[str] = list(job.requirement_lines) + [job_description]
    idx = {}
    for i, c in enumerate(cands):
        for v in ("blind", "raw", "swap"):
            chunks = skills.chunk_resume(c[v])
            idx[(i, v)] = (len(texts), len(texts) + len(chunks))
            texts += chunks
            idx[(i, v, "full")] = len(texts)
            texts.append(c[v][:4000])
    vecs, backend = embed(texts)
    n_req = len(job.requirement_lines)
    req_v, jd_v = vecs[:n_req], vecs[n_req]

    def semantic(i, v):
        a, b = idx[(i, v)]
        sim = cosine_matrix(req_v, vecs[a:b])           # (requirements x chunks) semantic search
        req_sim = float(sim.max(axis=1).mean())
        doc_sim = float(vecs[idx[(i, v, "full")]] @ jd_v)
        return _rescale(0.6 * req_sim + 0.4 * doc_sim, backend), sim, (a, b)

    # ---- score every candidate ----
    rows = []
    for i, c in enumerate(cands):
        sem, sim, (a, b) = semantic(i, "blind")
        profile = llm.extract_resume_profile(c["blind"])          # LLM sees redacted text only
        sk, mr, xr, mp, xp = skill_score(job, profile.skills)
        ex = experience_score(job, profile.years_experience)
        final = combine(sem, sk, ex)

        # evidence: best resume line per job requirement (top 3 strongest)
        ev = []
        for r in range(n_req):
            j = int(sim[r].argmax())
            ev.append(Evidence(requirement=job.requirement_lines[r],
                               best_resume_snippet=texts[a + j][:220],
                               similarity=round(float(sim[r, j]), 3)))
        ev.sort(key=lambda e: -e.similarity)

        # deterministic audit probe (rule-based profile, model-agnostic)
        def probe(v):
            s_sem, _, _ = semantic(i, v)
            sks = skills.extract_skills(c[v])
            return combine(s_sem, skill_score(job, sks)[0],
                           experience_score(job, skills.estimate_years(c[v])))
        rows.append({"c": c, "sem": sem, "sk": sk, "ex": ex, "final": final, "profile": profile,
                     "mr": mr, "xr": xr, "mp": mp, "xp": xp, "ev": ev,
                     "p_blind": probe("blind"), "p_raw": probe("raw"), "p_swap": probe("swap")})

    rows.sort(key=lambda r: -r["final"])
    rank_raw = {id(r): k + 1 for k, r in enumerate(sorted(rows, key=lambda r: -r["p_raw"]))}
    rank_blind = {id(r): k + 1 for k, r in enumerate(sorted(rows, key=lambda r: -r["p_blind"]))}

    results = []
    for k, r in enumerate(rows, 1):
        expl, tips = llm.explain_and_improve(job, r["profile"], r["mr"], r["xr"], r["xp"], r["final"],
                                             [e.model_dump() for e in r["ev"]])
        cf = r["p_swap"] - r["p_raw"]
        results.append(CandidateResult(
            rank=k, candidate=r["c"]["name"],
            scores=ScoreBreakdown(semantic=round(r["sem"], 3), skills=round(r["sk"], 3),
                                  experience=round(r["ex"], 3), final=round(r["final"], 1)),
            profile=r["profile"], matched_required=r["mr"], matched_preferred=r["mp"],
            missing_required=r["xr"], missing_preferred=r["xp"], evidence=r["ev"][:3],
            explanation=expl, improvements=tips,
            bias=BiasAudit(flagged_terms=r["c"]["flagged"], score_with_pii=round(r["p_raw"], 1),
                           score_blind=round(r["p_blind"], 1),
                           score_delta=round(r["p_blind"] - r["p_raw"], 1),
                           counterfactual_gender_delta=round(cf, 1),
                           rank_with_pii=rank_raw[id(r)], rank_blind=rank_blind[id(r)],
                           verdict=bias.verdict(cf, r["c"]["flagged"]))))
    if top_k:
        results = results[:top_k]

    max_cf = max((abs(x.bias.counterfactual_gender_delta) for x in results), default=0)
    shifted = sum(x.bias.rank_with_pii != x.bias.rank_blind for x in results)
    fallback = (f"Ranking used redacted resumes. Max gender-swap score change: {max_cf:.1f} pts; "
                f"{shifted}/{len(results)} candidates would change rank if PII were left in. "
                + ("No gender sensitivity detected." if max_cf < 1 else "Review flagged candidates."))
    summary = llm.fairness_summary([x.bias.model_dump() for x in results]) or fallback

    return RankResponse(job=job, llm_used=llm.llm_available(), llm_name=llm.llm_name(), embedding_backend=backend,
                        weights={"semantic": config.W_SEMANTIC, "skills": config.W_SKILLS,
                                 "experience": config.W_EXPERIENCE},
                        results=results, fairness_summary=summary)
