# 🎬 Audience Pulse

Read a room full of audience reviews in one pass. Paste a batch of reviews for a
film, show, or any title and get an instant pulse score, sentiment split, and the
harshest reviews surfaced first.

> **Author:** Adarsh Raj
> **Type:** End-to-end NLP project (deep learning) + interactive web app
> **Model:** Embedding → SimpleRNN → Dense, trained on 50,000 labelled movie reviews

---

## 🖥️ Try the App

**[Live demo →](#)** _(Render URL added once deployed)_

```bash
git clone https://github.com/adarshraj18/audience-pulse.git
cd audience-pulse
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` and paste in a batch of reviews (one per line, or
separated by a blank line). There's a "Try a sample batch" button if you just want
to see it work.

## 📌 Problem

A single review's sentiment is a curiosity. What a content or marketing team
actually needs before a launch, a re-release, or a mid-campaign check-in is a read
on a *batch* of audience reaction at once: is the room mostly positive, how split
is opinion, and which reviews are worth reading first instead of scrolling through
hundreds of them. Audience Pulse turns a block of pasted reviews into that summary
in one request.

## 🧠 Approach

1. **Model.** A small recurrent network (`Embedding → SimpleRNN → Dense`) trained
   from scratch on the [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
   (50,000 IMDB reviews, balanced positive/negative), via `tensorflow.keras.datasets.imdb`.
   See [`training/train.py`](training/train.py).
2. **Serving without TensorFlow.** Training happens once, offline, in Keras. The
   trained weights are then exported to plain NumPy arrays (`model/weights.npz`),
   and the deployed app reimplements the forward pass (embedding lookup, the
   RNN's step-by-step recurrence, the sigmoid output) in about 60 lines of NumPy
   (`app/model.py`). This is checked against the real Keras model in
   [`tests/test_model_matches_keras.py`](tests/test_model_matches_keras.py)
   (agreement within 1e-4), so it's a verified equivalent, not an approximation.
   The payoff: the deployed app needs only `fastapi`, `uvicorn`, and `numpy`, with
   no ~600MB TensorFlow install and fast cold starts.

   The same forward pass also has a third implementation, in vanilla JavaScript
   (`app/static/model.js` and `text.js`/`analyze.js`), verified against the NumPy
   version by [`scripts/verify_js_model.mjs`](scripts/verify_js_model.mjs). It's
   not used by the live deployment (the FastAPI backend does the scoring), but
   it's a fully working, tested alternative for anyone who'd rather run this as
   a pure static site with no backend at all: swap `index.html`'s script tag
   from `app.js` to [`app.offline.js`](app/static/app.offline.js), and nothing
   pasted into the app ever leaves the visitor's browser.
3. **Batch analysis.** [`app/analyze.py`](app/analyze.py) splits pasted text into
   individual reviews, scores each one, and aggregates a pulse score (share of
   reviews reading positive), the positive/negative split, and the five most
   negative reviews.
4. **Confidence signal.** Each review also gets a *coverage* score: the share of
   its words the model actually recognises from its 10,000-word vocabulary. A
   review full of rare words, slang, or non-English text gets flagged
   "low confidence" in the UI rather than presented as an equally trustworthy
   score. Sentiment models are frequently shipped without any signal for when
   they're guessing outside their training distribution, so this is a small,
   honest attempt to expose that.

### A design note worth being upfront about

The standard SimpleRNN activation is `tanh`, not `relu`. `relu` is unbounded and
noticeably less stable once you unroll it across the ~500-token sequences these
reviews are padded to, since nothing keeps the hidden state from growing across
timesteps. `tanh` is used here for that reason.

Even so, a plain SimpleRNN is a genuinely simple architecture, and it has a real,
known weakness: on longer reviews, its ability to carry signal across the full
sequence degrades (a form of the vanishing gradient problem), so a long,
subtly-worded review can occasionally get read as the opposite sentiment. This is
why the app surfaces a confidence signal rather than pretending every score is
equally reliable, and why the README says so instead of hiding it. An LSTM or GRU
would handle this better, at the cost of the model being much less readable as a
learning project. Trading some accuracy for a model whose recurrence you can
actually read top to bottom in `app/model.py` is a deliberate choice, not an
oversight.

### Test accuracy

**76.6% test accuracy, 0.51 test loss**, trained for up to 10 epochs with early
stopping (`patience=3` on validation loss). This is the actual result of the last
training run (see [`model/metrics.json`](model/metrics.json)), reported as-is and
not cherry-picked. A plain SimpleRNN tops out well short of what an LSTM/GRU or a
transformer-based model would get on this dataset (90%+ is common); see the design
note above for why that trade-off is intentional here.

## 🛠️ Project Structure

```
.
├── app/
│   ├── main.py            # FastAPI app: /api/analyze, /api/health, serves the frontend
│   ├── analyze.py          # Batch splitting + pulse/aggregate logic
│   ├── model.py            # NumPy-only inference (no TensorFlow at request time)
│   ├── text.py             # Tokeniser / encoder, matches the IMDB vocabulary
│   └── static/              # Frontend: index.html, styles.css, app.js (no build step)
│       ├── app.js            # Default entry point: calls the server's /api/analyze
│       ├── app.offline.js     # Alternate entry point: runs inference in-browser instead
│       ├── model.js           # Client-side inference (JS port of app/model.py)
│       ├── text.js            # Tokeniser / encoder (JS port of app/text.py)
│       ├── analyze.js         # Batch analysis (JS port of app/analyze.py)
│       └── model/               # Browser-loadable weights, used only by app.offline.js
├── training/
│   ├── train.py              # Trains the model, exports weights.npz + vocab.json
│   ├── export_web_weights.py  # Exports the browser (.bin) version, for app.offline.js
│   └── requirements.txt       # Extra deps needed only to retrain (TensorFlow, pytest)
├── model/
│   ├── weights.npz           # Trained weights, plain NumPy arrays
│   ├── vocab.json            # Word to id vocabulary used at both train and serve time
│   └── metrics.json          # Test accuracy/loss from the last training run
├── scripts/
│   └── verify_js_model.mjs    # Confirms app.offline.js's model.js matches app/model.py
├── tests/                     # Unit tests + NumPy-vs-Keras equivalence check
├── render.yaml                 # Render Blueprint: one-click deploy from this repo
├── Dockerfile                  # Deployment image (used for the live demo)
└── requirements.txt            # Runtime deps for the deployed app only
```

## 🚀 Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/adarshraj18/audience-pulse.git
cd audience-pulse

# 2. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install runtime dependencies (no TensorFlow needed to just run the app)
pip install -r requirements.txt

# 4. Launch
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

### Retraining the model

Only needed if you want to change the architecture or retrain from scratch:

```bash
pip install -r training/requirements.txt
python training/train.py
```

This downloads the IMDB dataset via Keras, trains the model, and overwrites
`model/weights.npz`, `model/vocab.json`, and `model/metrics.json`. If you're
also using the offline/static entry point (`app.offline.js`), re-export its
browser weights too:

```bash
python training/export_web_weights.py
```

### Running tests

```bash
# Python: unit tests + Keras-vs-NumPy equivalence
pip install -r training/requirements.txt  # pulls in TensorFlow + pytest
pytest tests/

# JavaScript: NumPy-vs-browser equivalence, only relevant if you use app.offline.js
node scripts/verify_js_model.mjs
```

## 📦 Deployment

The live demo runs the FastAPI app in a Docker container on
[Render](https://render.com)'s free tier, deployed straight from this GitHub
repo via [`render.yaml`](render.yaml) (a Render Blueprint): connect the repo
on Render, and it builds [`Dockerfile`](Dockerfile) and deploys automatically
on every push to `main`. The free tier spins the service down after 15 minutes
without traffic, so the first request after a quiet period can take 30-50
seconds while it wakes back up.

The model runs server-side here, scored by the same NumPy implementation
(`app/model.py`) that's verified against Keras. TensorFlow never ships to
production either way: the Docker image only installs
[`requirements.txt`](requirements.txt) (`fastapi`, `uvicorn`, `pydantic`,
`numpy`).

## 🔭 Next Steps

- Support pasting a CSV/export directly instead of plain text.
- Swap the RNN for a small LSTM/GRU as a documented "v2" comparison, to make the
  vanishing-gradient trade-off concrete rather than just described.
- Per-review keyword highlighting (which words pulled the score up or down).

## 📄 License

Released under the [MIT License](LICENSE). The training data is the Stanford
Large Movie Review Dataset (Maas et al., ACL 2011), used here for a non-commercial
educational/portfolio project via `tensorflow.keras.datasets.imdb`; see the
[dataset's terms](https://ai.stanford.edu/~amaas/data/sentiment/) for reuse
outside that context.

---

Built by **Adarsh Raj**.
