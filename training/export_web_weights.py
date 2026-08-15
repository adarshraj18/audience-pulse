"""Export the trained model as browser-loadable artifacts.

The deployed app is a static site (no Python server at request time), so the
weights need a format a browser can `fetch()` directly: raw Float32 binary
blobs, one per weight tensor, plus a small JSON manifest describing their
shapes and the vocabulary.

Run after training/train.py:

    .venv/bin/python training/export_web_weights.py

Writes to app/static/model/ (embedding.bin, rnn_kernel.bin, rnn_recurrent.bin,
rnn_bias.bin, dense_w.bin, dense_b.bin, manifest.json, vocab.json), and a
fixtures.json used by scripts/verify_js_model.mjs to confirm the JavaScript
forward pass (app/static/model.js) produces the same scores as this project's
NumPy implementation (app/model.py) for a fixed set of reviews.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
WEB_MODEL_DIR = ROOT / "app" / "static" / "model"

sys.path.insert(0, str(ROOT))
from app.model import SentimentModel  # noqa: E402
from app.text import Encoder  # noqa: E402

FIXTURE_REVIEWS = [
    "This movie was fantastic! The acting was great and the plot was thrilling.",
    "Waste of time. Terrible acting, boring plot, I want my two hours back.",
    "This was one of the best films I've seen all year. The pacing never let up.",
    "It was fine. Not great, not terrible, just an average watch.",
    "asdkjaslkdj qwoiue not a real review at all zzz",
]


def write_bin(array: np.ndarray, path: Path) -> None:
    array.astype(np.float32).tofile(path)


def main() -> None:
    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weights = np.load(MODEL_DIR / "weights.npz")

    write_bin(weights["embedding"], WEB_MODEL_DIR / "embedding.bin")
    write_bin(weights["rnn_kernel"], WEB_MODEL_DIR / "rnn_kernel.bin")
    write_bin(weights["rnn_recurrent"], WEB_MODEL_DIR / "rnn_recurrent.bin")
    write_bin(weights["rnn_bias"], WEB_MODEL_DIR / "rnn_bias.bin")
    write_bin(weights["dense_w"], WEB_MODEL_DIR / "dense_w.bin")
    write_bin(weights["dense_b"], WEB_MODEL_DIR / "dense_b.bin")

    vocab_size, embedding_dim = weights["embedding"].shape
    units = weights["rnn_bias"].shape[0]

    vocab_payload = json.loads((MODEL_DIR / "vocab.json").read_text(encoding="utf-8"))
    max_len = vocab_payload["max_len"]

    manifest = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "units": units,
        "max_len": max_len,
    }
    with open(WEB_MODEL_DIR / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    with open(WEB_MODEL_DIR / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab_payload, fh)

    # Fixture scores from the already-verified NumPy path, so the JS port
    # (app/static/model.js) can be checked against a known-good result
    # without needing TensorFlow in the loop at all.
    numpy_model = SentimentModel(MODEL_DIR / "weights.npz")
    encoder = Encoder.load(MODEL_DIR / "vocab.json")
    fixtures = [
        {"text": review, "score": numpy_model.predict(encoder.encode(review))}
        for review in FIXTURE_REVIEWS
    ]
    with open(WEB_MODEL_DIR / "fixtures.json", "w", encoding="utf-8") as fh:
        json.dump(fixtures, fh, indent=2)

    total_bytes = sum(
        (WEB_MODEL_DIR / name).stat().st_size
        for name in ["embedding.bin", "rnn_kernel.bin", "rnn_recurrent.bin", "rnn_bias.bin", "dense_w.bin", "dense_b.bin"]
    )
    print(f"Wrote browser model artifacts to {WEB_MODEL_DIR}/ ({total_bytes / 1_000_000:.1f} MB of weights)")


if __name__ == "__main__":
    main()
