"""Tests for the batch analyzer's splitting and aggregation logic.

Uses fake model/encoder doubles with fixed scores so these tests check the
*aggregation math* (pulse score, split counts, harshest ranking), not the
trained model's actual predictions; that equivalence is covered separately
in test_model_matches_keras.py.
"""

import pytest

from app.analyze import Analyzer, split_reviews


class ScoreMapEncoder:
    """Encoder that tags each review with itself so FakeModel can look it up."""

    def encode(self, text):
        return text

    def coverage(self, text):
        return 1.0 if text else 0.0


class ScoreMapModel:
    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text

    def predict(self, tagged_text):
        return self.scores_by_text[tagged_text]


def test_split_reviews_blank_line_separated():
    text = "Loved it.\n\nHated it.\n\nIt was fine."
    assert split_reviews(text) == ["Loved it.", "Hated it.", "It was fine."]


def test_split_reviews_one_per_line_fallback():
    text = "Loved it.\nHated it.\nIt was fine."
    assert split_reviews(text) == ["Loved it.", "Hated it.", "It was fine."]


def test_split_reviews_single_review_no_blank_lines():
    assert split_reviews("Just one review, no line breaks at all.") == [
        "Just one review, no line breaks at all."
    ]


def test_split_reviews_ignores_blank_input():
    assert split_reviews("   \n\n  ") == []


def test_analyze_batch_computes_pulse_and_split():
    scores = {"great": 0.9, "bad": 0.1, "meh": 0.4}
    analyzer = Analyzer(ScoreMapModel(scores), ScoreMapEncoder())

    report = analyzer.analyze_batch("great\n\nbad\n\nmeh")

    assert report.positive_count == 1  # only "great" > 0.5
    assert report.negative_count == 2
    assert report.pulse_score == pytest.approx(33.3, abs=0.1)
    assert [r.text for r in report.harshest][:2] == ["bad", "meh"]


def test_analyze_batch_empty_input_returns_no_reviews():
    analyzer = Analyzer(ScoreMapModel({}), ScoreMapEncoder())
    report = analyzer.analyze_batch("")
    assert report.reviews == []
    assert report.pulse_score == 0.0
