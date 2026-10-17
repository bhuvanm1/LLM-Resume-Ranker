"""Skill taxonomy + rule-based extraction (the offline fallback for the LLM)."""
import re
from datetime import date

# canonical skill -> aliases (all lowercase)
TAXONOMY: dict[str, list[str]] = {
    "Python": ["python", "py"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "node.js", "nodejs"],
    "TypeScript": ["typescript"],
    "SQL": ["sql", "postgresql", "postgres", "mysql", "sqlite"],
    "NoSQL": ["nosql", "mongodb", "cassandra", "dynamodb", "redis"],
    "Machine Learning": ["machine learning", "ml", "scikit-learn", "sklearn", "xgboost"],
    "Deep Learning": ["deep learning", "neural networks", "cnn", "rnn"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "keras"],
    "NLP": ["nlp", "natural language processing", "text mining"],
    "LLMs": ["llm", "llms", "large language models", "gpt", "claude", "openai", "langchain", "prompt engineering"],
    "Embeddings": ["embeddings", "embedding", "vector search", "semantic search", "faiss", "pinecone", "chromadb"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "REST APIs": ["rest api", "rest apis", "restful", "api development", "flask", "django"],
    "FastAPI": ["fastapi"],
    "Docker": ["docker", "containers", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "CI/CD": ["ci/cd", "jenkins", "github actions", "gitlab ci"],
    "Git": ["git", "github", "gitlab"],
    "Data Analysis": ["data analysis", "pandas", "numpy", "data analytics"],
    "Spark": ["spark", "pyspark"],
    "Airflow": ["airflow"],
    "React": ["react", "react.js"],
    "MLOps": ["mlops", "mlflow", "model deployment", "model monitoring"],
    "Statistics": ["statistics", "statistical modeling", "hypothesis testing", "a/b testing"],
    "Linux": ["linux", "bash", "shell scripting"],
    "Agile": ["agile", "scrum", "kanban"],
    "Communication": ["communication", "stakeholder management", "presentation skills"],
    "Leadership": ["leadership", "mentoring", "team lead", "led a team"],
}

# "FastAPI" is a more specific alias of REST APIs; keep both canonical names addressable.
_PATTERNS = {
    canon: re.compile(r"(?<![\w+#.])(" + "|".join(re.escape(a) for a in sorted(al, key=len, reverse=True)) + r")(?![\w+#])", re.I)
    for canon, al in TAXONOMY.items()
}


def extract_skills(text: str) -> list[str]:
    return [c for c, p in _PATTERNS.items() if p.search(text)]


def canonicalize(skill: str) -> str:
    """Map any free-text skill (e.g. from the LLM) onto the taxonomy when possible."""
    s = skill.strip().lower()
    for canon, aliases in TAXONOMY.items():
        if s == canon.lower() or s in aliases:
            return canon
    return skill.strip()


_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I)
_RANGE = re.compile(r"((?:19|20)\d{2})\s*[-–—to]+\s*((?:19|20)\d{2}|present|current|now)", re.I)


def estimate_years(text: str) -> float:
    """Explicit '5 years' claims win; otherwise sum employment date ranges."""
    explicit = [int(m) for m in _YEARS.findall(text) if int(m) <= 40]
    spans = []
    for a, b in _RANGE.findall(text):
        end = date.today().year if b.lower() in ("present", "current", "now") else int(b)
        if end >= int(a):
            spans.append((int(a), end))
    span_years = 0.0
    if spans:
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        span_years = float(min(hi - lo, 40))
    return float(max(explicit + [span_years] or [0]))


_TITLE = re.compile(
    r"\b((?:senior|junior|lead|principal|staff)?\s*(?:machine learning|ml|data|software|backend|frontend|full[- ]stack|devops|ai|nlp|research)?\s*"
    r"(?:engineer|scientist|developer|analyst|architect|manager|intern))\b", re.I)


def extract_titles(text: str) -> list[str]:
    seen, out = set(), []
    for m in _TITLE.findall(text):
        t = " ".join(m.split()).title()
        if t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return out[:5]


_EDU = re.compile(r"(ph\.?d|doctorate|master'?s?|m\.?s\.?c?|m\.?tech|mba|bachelor'?s?|b\.?s\.?c?|b\.?e\.?|b\.?tech)\b[^\n.]{0,60}", re.I)


def extract_education(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0).strip() for m in _EDU.finditer(text)))[:3]


def split_jd_sections(jd: str) -> tuple[str, str]:
    """Return (required_text, preferred_text) based on common headings."""
    m = re.search(r"(preferred|nice[- ]to[- ]have|bonus|good to have|plus)", jd, re.I)
    if not m:
        return jd, ""
    return jd[: m.start()], jd[m.start():]


_BULLET = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")


def requirement_lines(jd: str) -> list[str]:
    """Prefer bullet lines (typical requirement format); else any sentence-length line."""
    raw = [l for l in jd.splitlines() if l.strip()][1:]  # first line = title
    bullets = [_BULLET.sub("", l).strip() for l in raw if _BULLET.match(l)]
    pool = bullets if len(bullets) >= 3 else [l.strip() for l in raw]
    lines = [l for l in pool if 30 <= len(l) <= 220]
    return lines[:15] or [jd[:220]]


def chunk_resume(text: str) -> list[str]:
    parts = [re.sub(r"^[\s\-•*]+", "", l).strip() for l in text.splitlines()]
    chunks = [p for p in parts if len(p) >= 25]
    return chunks[:80] or [text[:500]]
