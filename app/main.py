"""Audience Pulse API and static frontend host.

The deployed app (app/static/) runs inference entirely in the browser via
model.js, so this FastAPI service isn't on the critical path for the live
demo. It exists for two reasons:

1. It serves app/static/ as-is, which is convenient for local development
   (`uvicorn app.main:app --reload`) and doubles as an optional Docker
   deployment target, since the static bundle works unmodified either way.
2. /api/analyze exposes the same, separately-tested batch analysis as a
   plain JSON API, for anyone who wants programmatic access without
   shipping the model to a client. Worth knowing: this loop is per-review
   and unbatched, which is fine on a real CPU but noticeably slow on
   constrained free-tier hosting (single review took ~8s on Render's free
   plan). That's exactly why the live deployment scores in the browser
   instead of calling this endpoint.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.analyze import Analyzer, MAX_REVIEWS_PER_REQUEST, PulseReport
from app.model import SentimentModel
from app.text import Encoder

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Audience Pulse", description="Batch sentiment reader for audience reviews")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model = SentimentModel(MODEL_DIR / "weights.npz")
_encoder = Encoder.load(MODEL_DIR / "vocab.json")
analyzer = Analyzer(_model, _encoder)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


class ReviewOut(BaseModel):
    text: str
    score: float
    sentiment: str
    coverage: float
    low_confidence: bool


class PulseOut(BaseModel):
    reviews: list[ReviewOut]
    pulse_score: float
    average_score: float
    positive_count: int
    negative_count: int
    harshest: list[ReviewOut]


def _to_out(report: PulseReport) -> PulseOut:
    to_review_out = lambda r: ReviewOut(**vars(r))  # noqa: E731
    return PulseOut(
        reviews=[to_review_out(r) for r in report.reviews],
        pulse_score=report.pulse_score,
        average_score=report.average_score,
        positive_count=report.positive_count,
        negative_count=report.negative_count,
        harshest=[to_review_out(r) for r in report.harshest],
    )


@app.post("/api/analyze", response_model=PulseOut)
def analyze(payload: AnalyzeRequest) -> PulseOut:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Paste at least one review.")
    report = analyzer.analyze_batch(text)
    if not report.reviews:
        raise HTTPException(status_code=400, detail="Couldn't find any reviews in that text.")
    return _to_out(report)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "max_reviews_per_request": MAX_REVIEWS_PER_REQUEST}


# Mounted last and at the root, after the /api/* routes above, so it only
# ever handles paths those routes didn't already claim. html=True serves
# index.html for "/", matching how a static host (the deployed target)
# behaves, so local dev exercises the exact bundle that ships.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
