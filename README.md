# LLM-Powered Resume Ranking & Job Matching

Upload resumes + a job description → skills extracted → candidates ranked by semantic relevance →
explanation, missing-skill analysis, resume improvement tips, and a bias audit.

## Architecture
```
resumes/JD ─► parsing (PDF/DOCX/TXT)
            ─► bias.redact()  (blind screening: names, emails, phones, pronouns, age, marital, grad-year…)
            ─► llm.extract_*  (Claude structured output via forced tool-use; rule-based fallback)
            ─► embeddings     (sentence-transformers or TF-IDF fallback; cosine similarity)
            ─► semantic search: each JD requirement ↔ best resume line (also gives "evidence")
            ─► score = 0.45·semantic + 0.40·skill coverage + 0.15·experience fit
            ─► llm.explain_and_improve  (explanation + truthful improvement tips)
            ─► bias audit (raw vs blind score, gender-swap counterfactual, rank shift)
```
| File | Role |
|---|---|
| `app/main.py` | FastAPI REST API + serves UI |
| `app/ranker.py` | Pipeline orchestration & scoring |
| `app/llm.py` | Claude structured outputs + offline fallbacks |
| `app/embeddings.py` | Embeddings + cosine similarity |
| `app/skills.py` | Skill taxonomy, years/title/education extraction |
| `app/bias.py` | Redaction, gender-swap counterfactual, verdicts |
| `app/schemas.py` | Pydantic models (API + LLM output contract) |
| `static/index.html` | Web UI |

## 100% free mode (Ollama, no API key)
1. Install Ollama from https://ollama.com and make sure it is running.
2. `ollama pull llama3.2` and `ollama pull nomic-embed-text`
3. Leave `ANTHROPIC_API_KEY` empty. The app auto-detects Ollama and uses it for both the LLM and embeddings.
   Header of the results page shows which engine was used.

## Run in VS Code
1. Install Python 3.10+ and open this folder: **File → Open Folder → resume-ranker**.
2. Terminal → New Terminal, then:
   ```
   python -m venv .venv
   # Windows PowerShell:  .venv\Scripts\Activate.ps1      (if blocked: Set-ExecutionPolicy -Scope Process Bypass)
   # Windows cmd:         .venv\Scripts\activate.bat
   # macOS/Linux:         source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Ctrl+Shift+P → Python: Select Interpreter →** pick `.venv`.
4. (Optional) `copy .env.example .env` (mac/Linux: `cp`) and set `ANTHROPIC_API_KEY`. No key = offline mode.
5. Start the server: `uvicorn app.main:app --reload`
6. Open http://127.0.0.1:8000 (UI) or http://127.0.0.1:8000/docs (Swagger).
   Try `sample_data/job_description.txt` + the files in `sample_data/resumes/`.
7. Tests: `pytest -q`

Optional better embeddings: `pip install sentence-transformers` (set `EMBEDDING_BACKEND=auto`).

## API
- `POST /api/rank` – multipart: `resumes` (files), `job_description` (text) or `job_file`
- `POST /api/rank/json` – `{"job_description": "...", "resumes": [{"name": "A", "text": "..."}]}`
- `POST /api/extract-skills` – `{"text": "..."}`
- `GET /api/health`

curl example:
```
curl -X POST http://127.0.0.1:8000/api/rank \
  -F "job_description=<sample_data/job_description.txt" \
  -F "resumes=@sample_data/resumes/Priya_Sharma.txt" -F "resumes=@sample_data/resumes/Arjun_Mehta.txt"
```

## Bias-aware design & limits
Ranking and LLM calls only see redacted text. The audit re-scores with PII, and with genders swapped, with a
deterministic probe; nonzero deltas flag sensitivity. This reduces, not eliminates, bias: proxies
(school names, locations, employment gaps) remain, and a human must make the final decision.
