# 🎬 Audience Pulse

Read a room full of audience reviews in one pass. Search a movie, or paste a batch
of reviews for a film, show, or any title, and get an instant pulse score,
sentiment split, and the harshest reviews surfaced first.

> **Author:** Adarsh Raj
> **Type:** End-to-end NLP project (deep learning) + interactive web app
> **Model:** Embedding → SimpleRNN → Dense, trained on 50,000 labelled movie reviews

---

## 🖥️ Try the App

**[Live demo →](https://audience-pulse.onrender.com)**

```bash
git clone https://github.com/adarshraj18/audience-pulse.git
cd audience-pulse/app/static
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000`. There are two ways in: search a movie by name (pulls
its reviews from TMDB, needs a free API key, see below), or paste in a batch of
reviews yourself (one per line, or separated by a blank line; there's a "Try a
sample batch" button if you just want to see it work). No `pip install` needed
for either: analysis runs entirely in your browser, so serving the static files
is all that's required.

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
2. **Inference runs in the browser.** Training happens once, offline, in Keras.
   The trained weights are then exported to raw Float32 binaries a browser can
   `fetch()` directly (`app/static/model/*.bin`, via
   [`training/export_web_weights.py`](training/export_web_weights.py)), and the
   deployed app reimplements the forward pass (embedding lookup, the RNN's
   step-by-step recurrence, the sigmoid output) in vanilla JavaScript
   (`app/static/model.js`) that runs on the visitor's own device. A second,
   NumPy implementation of the same forward pass (`app/model.py`) backs an
   optional server-side API. Both are checked against each other and against
   Keras: [`tests/test_model_matches_keras.py`](tests/test_model_matches_keras.py)
   for Keras vs. NumPy, and [`scripts/verify_js_model.mjs`](scripts/verify_js_model.mjs)
   for NumPy vs. JavaScript, both agreeing within 1e-4. So what runs in a
   visitor's browser is a verified equivalent of the trained model, not an
   approximation.

   Why the browser and not a server: I first deployed this behind a FastAPI
   backend on Render's free tier, and a single review took roughly 8 seconds to
   score there, since free-tier CPUs are heavily throttled and this model's
   per-timestep loop isn't a great fit for that. The same forward pass in a
   visitor's browser scores in single-digit milliseconds, since it runs on
   their own, much less constrained, device. `app/main.py` still exposes
   `/api/analyze` as a plain JSON API for anyone who wants server-side scoring
   on infrastructure with real CPU; it just isn't what the live demo uses.
3. **Batch analysis.** [`app/analyze.py`](app/analyze.py) (and its browser
   counterpart, `app/static/analyze.js`) splits pasted text into individual
   reviews, scores each one, and aggregates a pulse score (share of reviews
   reading positive), the positive/negative split, and the five most negative
   reviews.
4. **Confidence signal.** Each review also gets a *coverage* score: the share of
   its words the model actually recognises from its 10,000-word vocabulary. A
   review full of rare words, slang, or non-English text gets flagged
   "low confidence" in the UI rather than presented as an equally trustworthy
   score. Sentiment models are frequently shipped without any signal for when
   they're guessing outside their training distribution, so this is a small,
   honest attempt to expose that.
5. **Two ways to get reviews in.** [`app/static/tmdb.js`](app/static/tmdb.js)
   is a small client for [The Movie Database (TMDB)](https://www.themoviedb.org)'s
   free API: search a title, pick the right match, and its reviews feed straight
   into the same analysis pipeline as pasted text. It runs entirely client-side
   too, using an API key each visitor provides and TMDB supports for exactly
   this browser-side pattern; the key is stored only in that browser's
   `localStorage` and never touches this app's non-existent server. Worth
   knowing up front: TMDB's review coverage is real but thin (a handful of
   reviews for a big release, often none for a smaller one), nowhere near what
   a site like Rotten Tomatoes shows on its own pages (which has no public
   API). For a deeper batch, pasting reviews copied from wherever you found
   them is still the more thorough option.

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
│   ├── main.py             # FastAPI app: optional /api/analyze JSON API, serves the frontend
│   ├── analyze.py           # Batch splitting + pulse/aggregate logic (Python)
│   ├── model.py             # NumPy-only inference (Python reference implementation)
│   ├── text.py              # Tokeniser / encoder, matches the IMDB vocabulary (Python)
│   └── static/               # The deployed app: index.html, styles.css, app.js
│       ├── app.js             # Default entry point: scores reviews in the browser
│       ├── app.server.js       # Alternate entry point: calls the server's /api/analyze
│       ├── model.js            # Client-side inference (JS port of app/model.py)
│       ├── text.js             # Tokeniser / encoder (JS port of app/text.py)
│       ├── analyze.js          # Batch analysis (JS port of app/analyze.py)
│       ├── tmdb.js             # TMDB API client for the "search a movie" mode
│       └── model/                # Browser-loadable weights, used by app.js
├── training/
│   ├── train.py               # Trains the model, exports weights.npz + vocab.json
│   ├── export_web_weights.py   # Exports the browser (.bin) version to app/static/model/
│   └── requirements.txt        # Extra deps needed only to retrain (TensorFlow, pytest)
├── model/
│   ├── weights.npz            # Trained weights, plain NumPy arrays
│   ├── vocab.json             # Word to id vocabulary used at both train and serve time
│   └── metrics.json           # Test accuracy/loss from the last training run
├── scripts/
│   └── verify_js_model.mjs     # Confirms app/static/model.js matches app/model.py
├── tests/                      # Python unit tests + NumPy-vs-Keras equivalence check
├── render.yaml                  # Render Blueprint: one-click deploy from this repo
├── Dockerfile                   # Runs app/main.py in a container (serves the same static bundle)
└── requirements.txt             # Runtime deps for app/main.py
```

## 🚀 Running Locally

The deployed app has no backend dependency, so running it locally just means
serving static files:

```bash
git clone https://github.com/adarshraj18/audience-pulse.git
cd audience-pulse/app/static
python3 -m http.server 8000
```

Then open `http://127.0.0.1:8000`. (A browser refuses to run ES modules or
fetch local files over a bare `file://` URL, hence the tiny local server; any
static file server works, this one just ships with Python.)

### Running the optional FastAPI backend

Not required to use the app, but useful for local development with autoreload,
or if you want `/api/analyze` as a plain JSON endpoint:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Retraining the model

Only needed if you want to change the architecture or retrain from scratch:

```bash
pip install -r training/requirements.txt
python training/train.py               # trains the model, writes model/weights.npz etc.
python training/export_web_weights.py  # exports the browser version to app/static/model/
```

### Running tests

```bash
# Python: unit tests + Keras-vs-NumPy equivalence
pip install -r training/requirements.txt  # pulls in TensorFlow + pytest
pytest tests/

# JavaScript: NumPy-vs-browser equivalence (no npm install needed, no dependencies)
node scripts/verify_js_model.mjs
```

## 📦 Deployment

The live demo is hosted on [Render](https://render.com)'s free tier, deployed
straight from this GitHub repo via [`render.yaml`](render.yaml) (a Render
Blueprint): connect the repo on Render, and it builds [`Dockerfile`](Dockerfile)
and redeploys automatically on every push to `main`. The free tier spins the
service down after 15 minutes without traffic, so the first request after a
quiet period can take 30-50 seconds while it wakes back up; after that, pages
load quickly since the app itself is a static bundle.

Scoring happens in the visitor's browser, not on Render's server (see the
design note in Approach above for why). That also means Render's free-tier CPU
constraints don't affect how the app actually feels to use once it's awake.

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
