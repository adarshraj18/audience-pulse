"""Text encoding for the Audience Pulse model.

The model is trained on the Keras IMDB corpus, which ships as pre-tokenised
integer sequences. To score a raw review at serving time we have to reproduce
that encoding exactly, otherwise the numbers we feed the network mean nothing.

The rules Keras uses (see `keras.datasets.imdb`):

    start_char = 1   every sequence begins with it
    oov_char   = 2   stands in for any word the model was not trained on
    index_from = 3   frequency ranks are shifted by 3 to make room for the above

So a word ranked 47th most frequent becomes id 50. Any word whose id lands at
or beyond the vocabulary size was never seen during training and must collapse
to `oov_char`, not to some arbitrary in-range id.

`vocab.json` is written by the training run and already stores the final,
shifted, vocabulary-filtered ids, so serving needs no TensorFlow at all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

START_ID = 1
OOV_ID = 2

# The IMDB corpus was tokenised on words, lowercased, with punctuation dropped.
# Apostrophes are kept because the vocabulary genuinely contains forms like
# "don't" and "it's".
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Split raw review text into the word tokens the vocabulary uses."""
    text = _HTML_BREAK_RE.sub(" ", text)
    return _TOKEN_RE.findall(text.lower())


class Encoder:
    """Turns review text into a fixed-length integer sequence."""

    def __init__(self, vocab: dict[str, int], max_len: int):
        self.vocab = vocab
        self.max_len = max_len

    @classmethod
    def load(cls, path: str | Path) -> "Encoder":
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(vocab=payload["vocab"], max_len=payload["max_len"])

    def encode(self, text: str) -> list[int]:
        """Encode text, then pre-pad or pre-truncate to `max_len`.

        Truncating from the front is deliberate: a review's verdict usually
        sits in its closing sentences, so when something has to be dropped we
        drop the opening. This matches `pad_sequences(..., truncating='pre')`
        used during training, so train and serve stay consistent.
        """
        ids = [START_ID] + [self.vocab.get(word, OOV_ID) for word in tokenize(text)]
        if len(ids) >= self.max_len:
            return ids[-self.max_len :]
        return [0] * (self.max_len - len(ids)) + ids

    def coverage(self, text: str) -> float:
        """Share of words the model actually recognises, in [0, 1].

        Surfaced in the UI as an honesty signal: a review packed with names,
        slang or non-English words scores low here, which means the prediction
        deserves less trust regardless of how confident the sigmoid looks.
        """
        words = tokenize(text)
        if not words:
            return 0.0
        known = sum(1 for word in words if word in self.vocab)
        return known / len(words)
