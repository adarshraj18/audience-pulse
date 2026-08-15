"""Confirms the NumPy inference path (app/model.py) matches Keras exactly.

This is what justifies removing TensorFlow from the deployed app: if this
test passes, the NumPy forward pass is not an approximation, it is the same
arithmetic. Requires TensorFlow, so it only runs in the training environment,
not in the lightweight deployment environment.
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

tf = pytest.importorskip("tensorflow", reason="only runs where TensorFlow is installed")

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from app.model import SentimentModel  # noqa: E402
from app.text import Encoder  # noqa: E402


def _rebuild_keras_model():
    from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
    from tensorflow.keras.models import Sequential

    weights = np.load(MODEL_DIR / "weights.npz")
    vocab_size, embedding_dim = weights["embedding"].shape
    units = weights["rnn_bias"].shape[0]
    max_len = json.loads((MODEL_DIR / "vocab.json").read_text())["max_len"]

    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            SimpleRNN(units),  # default activation is tanh, matching train.py
            Dense(1, activation="sigmoid"),
        ]
    )
    model.build(input_shape=(None, max_len))
    model.layers[0].set_weights([weights["embedding"]])
    model.layers[1].set_weights([weights["rnn_kernel"], weights["rnn_recurrent"], weights["rnn_bias"]])
    model.layers[2].set_weights([weights["dense_w"], weights["dense_b"]])
    return model


@pytest.mark.parametrize(
    "review",
    [
        "This movie was fantastic! The acting was great and the plot was thrilling.",
        "Waste of time. Terrible acting, boring plot, I want my two hours back.",
        "It was fine. Not great, not terrible, just an average watch.",
        "",
        "asdkjaslkdj qwoiue not a real review at all zzz",
    ],
)
def test_numpy_matches_keras(review):
    encoder = Encoder.load(MODEL_DIR / "vocab.json")
    numpy_model = SentimentModel(MODEL_DIR / "weights.npz")
    keras_model = _rebuild_keras_model()

    token_ids = encoder.encode(review)

    numpy_score = numpy_model.predict(token_ids)
    keras_score = float(keras_model.predict(np.array([token_ids]), verbose=0)[0][0])

    assert numpy_score == pytest.approx(keras_score, abs=1e-4)
