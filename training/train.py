"""Train the Audience Pulse sentiment model.

A small Embedding -> SimpleRNN -> Dense network trained on the Keras IMDB
movie review corpus (50,000 labelled reviews). This is intentionally a
simple, classic architecture: the goal of this project is a trustworthy,
understandable model behind a genuinely useful tool, not a state-of-the-art
NLP result.

Run from the project root:

    .venv/bin/python training/train.py

Produces two artifacts under model/:
    weights.npz   the trained embedding / RNN / dense weights, as plain NumPy
                  arrays, for the dependency-light inference path (app/model.py)
    vocab.json    the word -> id vocabulary and max sequence length, used to
                  encode raw review text the same way at train and serve time
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10_000  # keep only the N most frequent words
MAX_LEN = 500  # reviews are padded/truncated to this many tokens
EMBEDDING_DIM = 128
RNN_UNITS = 128
EPOCHS = 10
BATCH_SIZE = 32
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"


def build_model() -> Sequential:
    model = Sequential(
        [
            Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
            # tanh is SimpleRNN's standard activation, and matters over a
            # 500-token sequence: unlike tanh, relu has no upper bound, so its
            # activations can grow across timesteps instead of settling,
            # making training on long sequences noticeably less stable.
            SimpleRNN(RNN_UNITS, activation="tanh"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def export_vocab(word_index: dict[str, int]) -> None:
    """Keep only ids inside VOCAB_SIZE; everything else is OOV at serve time."""
    kept = {word: idx for word, idx in word_index.items() if idx + 3 < VOCAB_SIZE}
    # word_index is 0-based; the model's embedding table is shifted by 3
    # (0=pad, 1=start, 2=oov), matching keras.datasets.imdb's convention.
    shifted = {word: idx + 3 for word, idx in kept.items()}
    payload = {"vocab": shifted, "max_len": MAX_LEN}
    with open(MODEL_DIR / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def export_weights(model: Sequential) -> None:
    embedding_w = model.layers[0].get_weights()[0]
    rnn_kernel, rnn_recurrent, rnn_bias = model.layers[1].get_weights()
    dense_w, dense_b = model.layers[2].get_weights()
    np.savez(
        MODEL_DIR / "weights.npz",
        embedding=embedding_w,
        rnn_kernel=rnn_kernel,
        rnn_recurrent=rnn_recurrent,
        rnn_bias=rnn_bias,
        dense_w=dense_w,
        dense_b=dense_b,
    )


def main() -> None:
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)

    print(f"Loading IMDB dataset (top {VOCAB_SIZE} words)...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = build_model()
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}  |  Test loss: {loss:.4f}")

    print("Exporting NumPy weights and vocabulary...")
    export_weights(model)
    export_vocab(imdb.get_word_index())

    metrics = {"test_accuracy": round(float(accuracy), 4), "test_loss": round(float(loss), 4)}
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Done. Artifacts written to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
