import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").lower()
SBERT_MODEL = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")

W_SEMANTIC = float(os.getenv("W_SEMANTIC", 0.45))
W_SKILLS = float(os.getenv("W_SKILLS", 0.40))
W_EXPERIENCE = float(os.getenv("W_EXPERIENCE", 0.15))

# LLM provider: auto | ollama | anthropic | none   (auto: anthropic if key set, else ollama if running, else none)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
