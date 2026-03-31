"""
Semantic density HTTP service for HASHIRU (Option B – Service 2).
Run in the semantic_density conda env from experiment_code (or set PYTHONPATH to experiment_code).

  cd experiment_code
  uvicorn density_service:app --host 127.0.0.1 --port 8125

Expects POST /score with body: { "prompt", "responses", "samples?", "metadata?" }.
Returns { "density": [float], "reasons": {} }.

**metadata:** accepted for gateway compatibility. `metadata["sequence_logprobs"]` is not used here;
semantic density is mean pairwise cosine similarity of embeddings only. Log-probs affect entropy (8124).

The "loading weights" message refers to the embedding model (e.g. all-MiniLM-L6-v2), not your agent LLM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# Reduce HuggingFace/httpx log noise (404s for optional files like adapter_config.json are normal)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from semantic_metrics import compute_semantic_density_from_text_responses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Density Service")


class ScoreRequest(BaseModel):
    prompt: str
    responses: List[str] = Field(..., min_length=1)
    samples: Optional[List[List[str]]] = None
    metadata: Optional[Dict[str, Any]] = None


class DensityScoreResponse(BaseModel):
    density: List[float]
    reasons: Dict[str, Any] = {}


@app.post("/score", response_model=DensityScoreResponse)
def score(req: ScoreRequest) -> DensityScoreResponse:
    if req.samples is not None and len(req.samples) != len(req.responses):
        raise HTTPException(
            status_code=422,
            detail=f"samples length ({len(req.samples)}) must match responses length ({len(req.responses)})",
        )
    n_resp = len(req.responses)
    n_samples_per = [len(req.samples[i]) if req.samples else 0 for i in range(n_resp)]
    logger.info("density request: %s response(s), samples per response: %s", n_resp, n_samples_per)

    densities: List[float] = []
    for i, resp in enumerate(req.responses):
        samples_i = req.samples[i] if req.samples else []
        all_responses = [resp] + list(samples_i)
        n_strings = len(all_responses)
        if n_strings < 2:
            logger.info("density item %s: only %s string(s), need ≥2 → density=1.0 (single response)", i, n_strings)
            densities.append(1.0)
            continue
        try:
            d = compute_semantic_density_from_text_responses(all_responses)
            d_clamped = max(0.0, min(1.0, float(d)))
            logger.info("density item %s: %s strings, raw=%.4f, clamped=%.4f", i, n_strings, float(d), d_clamped)
            densities.append(d_clamped)
        except Exception as e:
            logger.exception("density item %s: computation failed: %s", i, e)
            densities.append(0.5)
    logger.info("density response: densities=%s", densities)
    return DensityScoreResponse(
        density=densities,
        reasons={"n": len(req.responses)},
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "density"}
