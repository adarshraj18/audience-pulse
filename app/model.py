"""Dependency-light inference for the Audience Pulse model.

`training/train.py` trains the network in Keras, then exports its weights as
plain NumPy arrays (model/weights.npz). This module reimplements the exact
same forward pass (embedding lookup, a SimpleRNN scan, a sigmoid output)
using only NumPy.

Why bother: TensorFlow is roughly 600MB and several seconds of cold-start
just to run a handful of matrix multiplies at inference time. A ~60-line
NumPy implementation makes the deployed app small and fast, and as a side
effect makes the one part of an RNN that's usually a black box (the
recurrence) fully readable.

`tests/test_model_matches_keras.py` checks this against the real Keras model
so the two never quietly drift apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SentimentModel:
    """SimpleRNN sentiment classifier, forward pass only, NumPy-only."""

    def __init__(self, weights_path: str | Path = MODEL_DIR / "weights.npz"):
        weights = np.load(weights_path)
        self.embedding = weights["embedding"]  # (vocab_size, embedding_dim)
        self.rnn_kernel = weights["rnn_kernel"]  # (embedding_dim, units) - input weights
        self.rnn_recurrent = weights["rnn_recurrent"]  # (units, units) - hidden-state weights
        self.rnn_bias = weights["rnn_bias"]  # (units,)
        self.dense_w = weights["dense_w"]  # (units, 1)
        self.dense_b = weights["dense_b"]  # (1,)
        self.units = self.rnn_bias.shape[0]

    def predict(self, token_ids: list[int]) -> float:
        """Run one padded/truncated token sequence through the network.

        Returns a probability in [0, 1]: how positive the model reads the
        review, matching the sigmoid output Keras would produce for the same
        input.
        """
        x = self.embedding[np.asarray(token_ids, dtype=np.int64)]  # (seq_len, embed_dim)

        h = np.zeros(self.units, dtype=np.float32)
        for step in x:
            # Keras' SimpleRNN: h_t = tanh(x_t . W_kernel + h_{t-1} . W_recurrent + b)
            pre_activation = step @ self.rnn_kernel + h @ self.rnn_recurrent + self.rnn_bias
            h = np.tanh(pre_activation)

        logit = h @ self.dense_w + self.dense_b
        return float(_sigmoid(logit)[0])
