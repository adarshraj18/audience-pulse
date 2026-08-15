# 🎬 Audience Pulse

Read a room full of audience reviews in one pass. Paste a batch of reviews for a
film, show, or any title and get an instant pulse score, sentiment split, and the
harshest reviews surfaced first.

> **Author:** Adarsh Raj
> **Type:** End-to-end NLP project (deep learning) + interactive web app
> **Model:** Embedding → SimpleRNN → Dense, trained on 50,000 labelled movie reviews

---

## 🖥️ Try the App

**[Live demo →](https://huggingface.co/spaces/adarshraj18/audience-pulse)**

```bash
git clone https://github.com/adarshraj18/audience-pulse.git
cd audience-pulse/app/static
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000` and paste in a batch of reviews (one per line, or
separated by a blank line). There's a "Try a sample batch" button if you just want
to see it work. No `pip install` needed for this: the whole app is static HTML/CSS/JS
plus the trained model's weights, and it runs entirely in your browser.

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
2. **Inference runs in the browser, not on a server.** Training happens once,
   offline, in Keras. The trained weights are exported twice: to plain NumPy
   arrays (`model/weights.npz`, for a Python reference implementation) and to
   raw Float32 binaries the browser can `fetch()` directly
   (`app/static/model/*.bin`, via `training/export_web_weights.py`). The forward
   pass itself (embedding lookup, the RNN's step-by-step recurrence, the sigmoid
   output) is implemented three times: once in Keras (training), once in NumPy
   (`app/model.py`), and once in vanilla JavaScript (`app/static/model.js`) that
   runs client-side. Each pair is checked against the other:
   [`tests/test_model_matches_keras.py`](tests/test_model_matches_keras.py) for
   Keras vs. NumPy, and [`scripts/verify_js_model.mjs`](scripts/verify_js_model.mjs)
   for NumPy vs. JavaScript, both agreeing within 1e-4. So the JS running in a
   visitor's browser is a verified equivalent of the trained model, not an
   approximation. The payoff: nothing you paste ever leaves your machine, there's
   no server to keep warm, and the whole app deploys as static files.
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
│   ├── main.py           # Optional FastAPI app: /api/analyze, /api/health, local dev server
│   ├── analyze.py         # Batch splitting + pulse/aggregate logic (Python)
│   ├── model.py           # NumPy-only inference (Python reference implementation)
│   ├── text.py            # Tokeniser / encoder, matches the IMDB vocabulary (Python)
│   └── static/             # The deployed app: fully static, this is what ships
│       ├── index.html
│       ├── styles.css
│       ├── app.js          # Wires everything below together, renders the UI
│       ├── model.js         # Client-side inference (JS port of app/model.py)
│       ├── text.js          # Tokeniser / encoder (JS port of app/text.py)
│       ├── analyze.js       # Batch analysis (JS port of app/analyze.py)
│       └── model/            # Browser-loadable weights, exported by training/export_web_weights.py
├── training/
│   ├── train.py             # Trains the model, exports weights.npz + vocab.json
│   ├── export_web_weights.py # Exports the browser (.bin) version of the trained model
│   └── requirements.txt      # Extra deps needed only to retrain (TensorFlow, pytest)
├── model/
│   ├── weights.npz          # Trained weights, plain NumPy arrays
│   ├── vocab.json           # Word to id vocabulary used at both train and serve time
│   └── metrics.json         # Test accuracy/loss from the last training run
├── scripts/
│   └── verify_js_model.mjs   # Confirms app/static/model.js matches app/model.py
├── tests/                    # Python unit tests + NumPy-vs-Keras equivalence check
├── Dockerfile                # Optional: runs the FastAPI app in a container
└── requirements.txt           # Runtime deps for the optional FastAPI app only
```

## 🚀 Running Locally

The deployed app has no backend, so running it locally just means serving
static files:

```bash
git clone https://github.com/adarshraj18/audience-pulse.git
cd audience-pulse/app/static
python3 -m http.server 8000
```

Then open `http://127.0.0.1:8000`. (A browser refuses to run ES modules or
fetch local files over a bare `file://` URL, hence the tiny local server;
any static file server works, this one just ships with Python.)

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

The live demo is a **static** [Hugging Face Space](https://huggingface.co/spaces)
(free tier, permanent URL): just the contents of `app/static/`, pushed to the
Space's own git repository.

```bash
# from the repo root, one-time setup
git remote add space https://huggingface.co/spaces/adarshraj18/audience-pulse

# whenever app/static/ changes
git subtree push --prefix app/static space main
```

`git subtree push` sends only the `app/static/` folder, as the root of the
Space's repo, so the Space never needs to know about the training code, tests,
or the optional FastAPI backend. The optional [`Dockerfile`](Dockerfile) is kept
for anyone who'd rather run the FastAPI version in a container elsewhere; it
isn't part of the live deployment.

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
