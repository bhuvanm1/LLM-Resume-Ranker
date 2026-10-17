from pathlib import Path
from fastapi.testclient import TestClient
from app import bias, skills
from app.main import app
from app.ranker import rank_candidates

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
JD = (SAMPLE / "job_description.txt").read_text()
RES = [(p.stem, p.read_text()) for p in sorted((SAMPLE / "resumes").glob("*.txt"))]


def test_skill_extraction():
    s = skills.extract_skills("Built REST APIs in Python, deployed with Docker on AWS")
    assert {"Python", "REST APIs", "Docker", "AWS"} <= set(s)


def test_redaction_removes_pii():
    out, flagged = bias.redact(RES[0][1], RES[0][0])
    assert "@" not in out and "Priya" not in out and "She " not in out
    assert flagged


def test_ranking_order_and_gaps():
    r = rank_candidates(JD, RES)
    names = [x.candidate for x in r.results]
    assert names[0] == "Priya_Sharma" and names[-1] in ("Arjun_Mehta", "Ananya_Iyer")
    assert r.results[0].scores.final > r.results[-1].scores.final
    assert r.results[-1].missing_required
    assert all(x.explanation and x.improvements for x in r.results)


def test_blind_ranking_is_gender_invariant():
    r = rank_candidates(JD, RES)
    assert all(abs(x.bias.counterfactual_gender_delta) < 5 for x in r.results)


def test_api_upload():
    c = TestClient(app)
    files = [("resumes", (n + ".txt", t.encode(), "text/plain")) for n, t in RES]
    resp = c.post("/api/rank", data={"job_description": JD}, files=files)
    assert resp.status_code == 200 and len(resp.json()["results"]) == len(RES)
    assert c.get("/api/health").json()["status"] == "ok"
