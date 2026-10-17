"""Embedding backends. Both return L2-normalised vectors, so
cosine similarity == dot product."""
import numpy as np
from . import config

_sbert = None


def _try_sbert():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer(config.SBERT_MODEL)
    return _sbert


def _ollama_embed(texts: list[str]) -> np.ndarray:
    import httpx
    out = []
    for i in range(0, len(texts), 64):
        r = httpx.post(f"{config.OLLAMA_URL}/api/embed", timeout=300,
                       json={"model": config.OLLAMA_EMBED_MODEL, "input": texts[i:i + 64]})
        r.raise_for_status()
        out += r.json()["embeddings"]
    m = np.asarray(out, dtype=float)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


def embed(texts: list[str]) -> tuple[np.ndarray, str]:
    """Embed all texts in ONE call (TF-IDF needs the whole batch to fit).
    Order in auto mode: Ollama -> sentence-transformers -> TF-IDF."""
    if config.EMBEDDING_BACKEND in ("auto", "ollama"):
        try:
            from . import llm
            if llm.ollama_running():
                return _ollama_embed(texts), f"ollama:{config.OLLAMA_EMBED_MODEL}"
            raise RuntimeError("Ollama not running")
        except Exception as e:
            if config.EMBEDDING_BACKEND == "ollama":
                raise RuntimeError(f"Ollama embeddings failed (did you run 'ollama pull {config.OLLAMA_EMBED_MODEL}'?): {e}")
    if config.EMBEDDING_BACKEND in ("auto", "sbert"):  # noqa
        try:
            model = _try_sbert()
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs), f"sentence-transformers:{config.SBERT_MODEL}"
        except Exception:
            if config.EMBEDDING_BACKEND == "sbert":
                raise
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
    m = vec.fit_transform(texts)
    return m.toarray(), "tfidf (local fallback)"


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity; rows are already normalised."""
    return a @ b.T
