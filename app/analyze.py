"""Batch review analysis: the actual feature behind Audience Pulse.

A single review's sentiment score is a curiosity. What a marketing or content
team actually needs is a read on a *batch* of audience reviews for one title
at once: the overall pulse, how split opinion is, and which reviews are worth
reading first. This module turns a block of pasted text into that summary.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.model import SentimentModel
from app.text import Encoder

POSITIVE_THRESHOLD = 0.5
LOW_CONFIDENCE_COVERAGE = 0.6  # below this, warn that the model recognised too few words
MAX_REVIEWS_PER_REQUEST = 200


@dataclass
class ReviewResult:
    text: str
    score: float
    sentiment: str
    coverage: float
    low_confidence: bool


@dataclass
class PulseReport:
    reviews: list[ReviewResult]
    pulse_score: float  # 0-100, share of reviews read as positive
    average_score: float  # mean raw model score, 0-1
    positive_count: int
    negative_count: int
    harshest: list[ReviewResult]  # most negative reviews, worth reading first


def split_reviews(raw_text: str) -> list[str]:
    """Split pasted text into individual reviews.

    Reviews are expected one-per-blank-line-separated-block (how people
    naturally paste a batch copied from a spreadsheet or review site). If the
    user pasted one review per single line instead, blank-line splitting
    yields one giant block, so fall back to splitting on single newlines.
    """
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    if len(blocks) > 1:
        return blocks
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return lines if lines else ([raw_text.strip()] if raw_text.strip() else [])


class Analyzer:
    def __init__(self, model: SentimentModel, encoder: Encoder):
        self.model = model
        self.encoder = encoder

    def analyze_one(self, text: str) -> ReviewResult:
        token_ids = self.encoder.encode(text)
        score = self.model.predict(token_ids)
        coverage = self.encoder.coverage(text)
        return ReviewResult(
            text=text,
            score=score,
            sentiment="positive" if score > POSITIVE_THRESHOLD else "negative",
            coverage=coverage,
            low_confidence=coverage < LOW_CONFIDENCE_COVERAGE,
        )

    def analyze_batch(self, raw_text: str) -> PulseReport:
        review_texts = split_reviews(raw_text)[:MAX_REVIEWS_PER_REQUEST]
        results = [self.analyze_one(text) for text in review_texts]

        positive_count = sum(1 for r in results if r.sentiment == "positive")
        negative_count = len(results) - positive_count
        average_score = sum(r.score for r in results) / len(results) if results else 0.0
        pulse_score = (positive_count / len(results) * 100) if results else 0.0

        harshest = sorted(results, key=lambda r: r.score)[:5]

        return PulseReport(
            reviews=results,
            pulse_score=round(pulse_score, 1),
            average_score=round(average_score, 4),
            positive_count=positive_count,
            negative_count=negative_count,
            harshest=harshest,
        )
