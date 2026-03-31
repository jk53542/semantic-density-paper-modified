import logging
import numpy as np

logger = logging.getLogger(__name__)

# Optional real sentence embeddings (install sentence-transformers for meaningful density)
_embed_model = None
_logged_iJIT_workaround = False

def _get_embed_model():
    global _embed_model, _logged_iJIT_workaround
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Using SentenceTransformer for semantic density embeddings (density values will be meaningful).")
        return _embed_model
    except Exception as e:
        err_str = str(e)
        logger.warning("sentence_transformers not available (%s); using fake embeddings (density will be unreliable).", e)
        if not _logged_iJIT_workaround and ("iJIT_NotifyEvent" in err_str or "undefined symbol" in err_str):
            _logged_iJIT_workaround = True
            logger.warning(
                "PyTorch/MKL symbol error. In semantic_density env try: conda install mkl=2024.0.0 "
                "or pip install torch sentence-transformers. See SETUP_SEMANTIC_ENVIRONMENTS.md."
            )
        return None

def embed_text(text: str, max_length: int = 512):
    """Embed text using real model if available, else deterministic fake for tests.
    max_length is ignored for encode() (newer sentence_transformers disallows it for some models); text is already truncated by caller.
    """
    model = _get_embed_model()
    if model is not None:
        emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float64)
    return embed_text_fake(text)

def compute_semantic_density_from_similarity_matrix(S):
    """
    Matches the implementation in the GitHub repo:
    experiment_code/semantic_metrics.py

    Semantic density = mean of off-diagonal cosine similarities.
    """
    n = S.shape[0]
    off_diag = S[np.triu_indices(n, k=1)]
    return float(np.mean(off_diag))


# -----------------------------------------------------------------------
# New helper: deterministic fake embedding generator for unit tests only
# -----------------------------------------------------------------------
def embed_text_fake(text: str) -> np.ndarray:
    """
    Produces a deterministic pseudo-embedding for a string.
    Allows the unit tests to simulate embeddings without a model.
    """
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=128)


def compute_similarity_matrix(embeddings):
    """
    Computes cosine similarity matrix.
    Follows what the repo does when computing similarity matrices.
    """
    embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def _truncate_to_words(text: str, max_words: int = 200) -> str:
    """Use the same chunk size as entropy so we compare comparable content; improves density for same-topic responses."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def compute_semantic_density_from_text_responses(responses, max_words: int = 200, max_seq_length: int = 256):
    """
    Embed → similarity matrix → density.
    Uses real sentence embeddings when sentence_transformers is available.
    Truncates each response to max_words so we compare comparable chunks.
    """
    using_real = _get_embed_model() is not None
    logger.info("compute_semantic_density: n_responses=%s, max_words=%s, using_real_embeddings=%s", len(responses), max_words, using_real)
    truncated = [_truncate_to_words(r, max_words) for r in responses]
    embeddings = [embed_text(t) for t in truncated]
    S = compute_similarity_matrix(embeddings)
    raw_d = compute_semantic_density_from_similarity_matrix(S)
    logger.info("compute_semantic_density: raw mean similarity (density)=%.4f", raw_d)
    return raw_d
