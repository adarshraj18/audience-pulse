"""Tests for the tokeniser/encoder. No TensorFlow required."""

from pathlib import Path

from app.text import Encoder, tokenize

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Fantastic! Truly great.") == ["fantastic", "truly", "great"]


def test_tokenize_strips_html_breaks():
    assert tokenize("Great film.<br />Would watch again.") == ["great", "film", "would", "watch", "again"]


def test_tokenize_keeps_apostrophes():
    assert tokenize("It's the best, don't miss it") == ["it's", "the", "best", "don't", "miss", "it"]


def test_encode_pads_short_sequences_on_the_left():
    encoder = Encoder(vocab={"hello": 10}, max_len=5)
    ids = encoder.encode("hello")
    assert len(ids) == 5
    assert ids[-1] != 0  # the last (most recent) token is real content
    assert ids.count(0) == 3  # 5 slots - start token - 1 word = 3 pad slots


def test_encode_truncates_from_the_front_when_too_long():
    encoder = Encoder(vocab={"a": 5, "b": 6, "c": 7}, max_len=3)
    ids = encoder.encode("a b c")
    # start token gets dropped first, then the oldest word, keeping the tail
    assert len(ids) == 3


def test_encode_maps_unknown_words_to_oov():
    encoder = Encoder(vocab={"known": 10}, max_len=4)
    ids = encoder.encode("unknownword known")
    assert 2 in ids  # OOV_ID
    assert 10 in ids


def test_coverage_all_known():
    encoder = Encoder(vocab={"great": 1, "film": 2}, max_len=10)
    assert encoder.coverage("great film") == 1.0


def test_coverage_partial():
    encoder = Encoder(vocab={"great": 1}, max_len=10)
    assert encoder.coverage("great unknownword") == 0.5


def test_coverage_empty_text_is_zero():
    encoder = Encoder(vocab={"great": 1}, max_len=10)
    assert encoder.coverage("") == 0.0


def test_trained_vocab_loads():
    encoder = Encoder.load(MODEL_DIR / "vocab.json")
    assert encoder.max_len == 500
    assert "the" in encoder.vocab
