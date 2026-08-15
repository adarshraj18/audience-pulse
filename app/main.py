"""Audience Pulse API.

A small FastAPI service that scores a batch of audience reviews for a title
and returns an overall pulse, the positive/negative split, and the harshest
reviews worth reading first. Serves the static frontend directly, so the
whole app is one process to deploy.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
