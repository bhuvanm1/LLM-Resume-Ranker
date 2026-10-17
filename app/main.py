"""REST API. Run:  uvicorn app.main:app --reload"""
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm, skills
from .parsing import extract_text
from .ranker import rank_candidates
from .schemas import RankRequest, RankResponse, ResumeProfile

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="LLM Resume Ranking & Job Matching", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_enabled": llm.llm_available(), "llm": llm.llm_name()}


class TextIn(BaseModel):
    text: str


@app.post("/api/extract-skills", response_model=ResumeProfile)
def extract(body: TextIn):
    """Structured skill/profile extraction for a single resume text."""
    return llm.extract_resume_profile(body.text)


@app.post("/api/rank/json", response_model=RankResponse)
def rank_json(req: RankRequest):
    if not req.resumes:
        raise HTTPException(400, "Provide at least one resume.")
    return rank_candidates(req.job_description, [(r.name, r.text) for r in req.resumes], req.top_k)


@app.post("/api/rank", response_model=RankResponse)
async def rank_upload(
    resumes: list[UploadFile] = File(...),
    job_description: str = Form(""),
    job_file: UploadFile | None = File(None),
    top_k: int | None = Form(None),
):
    """Multipart upload: resumes (PDF/DOCX/TXT) + job description (text or file)."""
    jd = job_description
    if job_file is not None and job_file.filename:
        jd = extract_text(job_file.filename, await job_file.read())
    if not jd.strip():
        raise HTTPException(400, "Job description is required (text or file).")
    parsed = []
    for f in resumes:
        text = extract_text(f.filename or "resume.txt", await f.read())
        if text.strip():
            parsed.append((Path(f.filename).stem, text))
    if not parsed:
        raise HTTPException(400, "No readable resumes uploaded.")
    return rank_candidates(jd, parsed, top_k)
