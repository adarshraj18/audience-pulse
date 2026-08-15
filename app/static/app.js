// Audience Pulse frontend: loads the model directly into the browser and
// runs every prediction locally (model.js + text.js + analyze.js), renders
// the pulse meter, the sentiment split, and the review lists. No framework,
// no build step, and nothing pasted or searched ever leaves the visitor's
// machine except the TMDB lookup itself (tmdb.js), which goes straight from
// their browser to TMDB using this app's shared API key (see tmdb.js for why
// that's fine to ship publicly).
//
// Two ways in: search a movie (fetches its reviews from TMDB) or paste
// reviews from anywhere. Both end up calling Analyzer.analyzeTexts and the
// same render() below.
//
// This runs client-side rather than calling the server's /api/analyze
// (app/main.py has that route, and app.server.js calls it) because free-tier
// hosting CPUs are too slow for this workload server-side: a single review
// took ~8 seconds on Render's free plan, versus single-digit milliseconds
// running the same forward pass in the visitor's own browser. Verified
// against the trained model by scripts/verify_js_model.mjs.

import { SentimentModel } from "./model.js";
import { Encoder } from "./text.js";
import { Analyzer } from "./analyze.js";
import { searchMovies, fetchReviews, DEFAULT_API_KEY } from "./tmdb.js";

const SAMPLE_BATCH = `This was one of the best films I've seen all year. The pacing never let up and the ending actually earned its emotional weight.

Honestly disappointing. The trailer promised so much more than this delivered, and the second act just drags.

A solid watch. Not perfect, but the lead performance carries it through some weaker plotting.

I walked out halfway through. Flat dialogue, no chemistry between the leads, and the twist was obvious from minute ten.

Genuinely surprised by how good this was. Smart script, great cinematography, and a soundtrack that fit every scene.

Mediocre at best. Feels like a rough draft of a much better movie that never got made.

Loved every minute. Took my whole family and everyone had a different favorite scene.

Overhyped. Fine for a background watch but forgettable within a day.`;

const els = {
  textarea: document.getElementById("reviews"),
  reviewCount: document.getElementById("reviewCount"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  analyzeLabel: document.getElementById("analyzeLabel"),
  spinner: document.getElementById("spinner"),
  sampleBtn: document.getElementById("sampleBtn"),
  errorBanner: document.getElementById("errorBanner"),
  results: document.getElementById("results"),
  pulseFigure: document.getElementById("pulseFigure"),
  meterFill: document.getElementById("meterFill"),
  pulseCaption: document.getElementById("pulseCaption"),
  positiveCount: document.getElementById("positiveCount"),
  negativeCount: document.getElementById("negativeCount"),
  totalCount: document.getElementById("totalCount"),
  splitGood: document.getElementById("splitGood"),
  splitBad: document.getElementById("splitBad"),
  splitGoodLabel: document.getElementById("splitGoodLabel"),
  splitBadLabel: document.getElementById("splitBadLabel"),
  harshList: document.getElementById("harshList"),
  allList: document.getElementById("allList"),
  showMoreBtn: document.getElementById("showMoreBtn"),
  themeToggle: document.getElementById("themeToggle"),
  modelStatus: document.getElementById("modelStatus"),
  tabSearch: document.getElementById("tabSearch"),
  tabPaste: document.getElementById("tabPaste"),
  searchPanel: document.getElementById("searchPanel"),
  pastePanel: document.getElementById("pastePanel"),
  movieQuery: document.getElementById("movieQuery"),
  movieSearchBtn: document.getElementById("movieSearchBtn"),
  searchSpinner: document.getElementById("searchSpinner"),
  searchLabel: document.getElementById("searchLabel"),
  movieResults: document.getElementById("movieResults"),
  searchStatus: document.getElementById("searchStatus"),
};

let analyzer = null;

// Using this entry point standalone (without index.html's #modelStatus
// element and the button's initial disabled state) still works; these
// status updates are just skipped if that markup isn't present.
function setModelStatus(text) {
  if (els.modelStatus) els.modelStatus.textContent = text;
}

async function loadModel() {
  try {
    const [model, encoder] = await Promise.all([SentimentModel.load("./model"), Encoder.load("./model/vocab.json")]);
    analyzer = new Analyzer(model, encoder);
    els.analyzeBtn.disabled = false;
    els.spinner.classList.remove("active");
    els.analyzeLabel.textContent = "Analyze pulse";
    setModelStatus("Model loaded, ready to analyze, entirely in your browser.");
    setTimeout(() => {
      if (els.modelStatus) els.modelStatus.style.display = "none";
    }, 2500);
  } catch (err) {
    els.analyzeLabel.textContent = "Model failed to load";
    setModelStatus("Couldn't load the model. Try reloading the page.");
    els.spinner.classList.remove("active");
  }
}

function showError(message) {
  els.errorBanner.textContent = message;
  els.errorBanner.classList.add("active");
}

function clearError() {
  els.errorBanner.classList.remove("active");
}

function countReviews() {
  const text = els.textarea.value;
  const blocks = text.split("\n\n").map((b) => b.trim()).filter(Boolean);
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const n = blocks.length > 1 ? blocks.length : lines.length;
  els.reviewCount.textContent = n ? `${n} review${n === 1 ? "" : "s"} detected` : "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function reviewCard(review) {
  const badgeClass = review.sentiment === "positive" ? "good" : "critical";
  const badgeIcon = review.sentiment === "positive" ? "▲" : "▼";
  const lowConfBadge = review.low_confidence
    ? `<span class="badge warning" title="The model recognised few of these words, read with caution">⚠ low confidence</span>`
    : "";
  const preview = review.text.length > 400 ? review.text.slice(0, 400) + "…" : review.text;
  return `
    <div class="review-card">
      <div class="review-meta">
        <span class="badge ${badgeClass}">${badgeIcon} ${review.sentiment}</span>
        ${lowConfBadge}
        <span class="review-score">score ${review.score.toFixed(2)}</span>
      </div>
      <p class="review-text">${escapeHtml(preview)}</p>
    </div>`;
}

function render(report) {
  els.pulseFigure.textContent = Math.round(report.pulse_score);
  els.meterFill.style.width = `${report.pulse_score}%`;
  els.pulseCaption.textContent =
    report.pulse_score >= 60
      ? "audience is largely positive"
      : report.pulse_score <= 40
      ? "audience is largely negative"
      : "opinion is split";

  els.positiveCount.textContent = report.positive_count;
  els.negativeCount.textContent = report.negative_count;
  els.totalCount.textContent = report.reviews.length;

  const total = report.reviews.length || 1;
  const goodPct = (report.positive_count / total) * 100;
  const badPct = 100 - goodPct;
  els.splitGood.style.width = `${goodPct}%`;
  els.splitBad.style.width = `${badPct}%`;
  els.splitGoodLabel.textContent = `${Math.round(goodPct)}% positive`;
  els.splitBadLabel.textContent = `${Math.round(badPct)}% negative`;

  els.harshList.innerHTML = report.harshest.map(reviewCard).join("");
  els.allList.innerHTML = report.reviews.map(reviewCard).join("");
  els.allList.classList.add("collapsed");
  els.showMoreBtn.style.display = report.reviews.length > 5 ? "block" : "none";
  els.showMoreBtn.textContent = "Show all reviews";

  els.results.classList.add("active");
  els.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyze() {
  const text = els.textarea.value.trim();
  clearError();
  if (!text) {
    showError("Paste at least one review first.");
    return;
  }
  if (!analyzer) {
    showError("The model is still loading, one moment.");
    return;
  }

  els.analyzeBtn.disabled = true;
  els.spinner.classList.add("active");
  els.analyzeLabel.textContent = "Analyzing…";

  try {
    // Yield to the browser first so the "Analyzing…" state actually paints
    // before the (synchronous, CPU-bound) inference loop runs.
    await new Promise((resolve) => setTimeout(resolve, 0));
    const report = analyzer.analyzeBatch(text);
    if (!report.reviews.length) {
      throw new Error("Couldn't find any reviews in that text.");
    }
    render(report);
  } catch (err) {
    showError(err.message || "Something went wrong analyzing that batch.");
  } finally {
    els.analyzeBtn.disabled = false;
    els.spinner.classList.remove("active");
    els.analyzeLabel.textContent = "Analyze pulse";
  }
}

// ---------- Mode tabs ----------

function switchTab(mode) {
  const isSearch = mode === "search";
  els.tabSearch.classList.toggle("active", isSearch);
  els.tabPaste.classList.toggle("active", !isSearch);
  els.tabSearch.setAttribute("aria-selected", String(isSearch));
  els.tabPaste.setAttribute("aria-selected", String(!isSearch));
  els.searchPanel.classList.toggle("hidden", !isSearch);
  els.pastePanel.classList.toggle("hidden", isSearch);
  clearError();
}

els.tabSearch.addEventListener("click", () => switchTab("search"));
els.tabPaste.addEventListener("click", () => switchTab("paste"));

// ---------- Movie search (TMDB) ----------
// Uses this app's shared DEFAULT_API_KEY (see tmdb.js), so there's no setup
// step: visitors can search the moment the page loads.

async function searchMovie() {
  const query = els.movieQuery.value.trim();
  clearError();
  els.searchStatus.textContent = "";
  els.movieResults.innerHTML = "";
  if (!query) {
    showError("Type a movie name first.");
    return;
  }

  els.movieSearchBtn.disabled = true;
  els.searchSpinner.classList.add("active");
  els.searchLabel.textContent = "Searching…";

  try {
    const movies = await searchMovies(query, DEFAULT_API_KEY);
    if (!movies.length) {
      els.searchStatus.textContent = `No matches for "${query}" on TMDB.`;
      return;
    }
    renderMovieResults(movies);
  } catch (err) {
    showError(err.message || "Search failed. Try again.");
  } finally {
    els.movieSearchBtn.disabled = false;
    els.searchSpinner.classList.remove("active");
    els.searchLabel.textContent = "Search";
  }
}

function renderMovieResults(movies) {
  els.movieResults.innerHTML = movies
    .map((m, i) => {
      const poster = m.posterUrl
        ? `<img class="movie-result-poster" src="${m.posterUrl}" alt="" />`
        : `<div class="movie-result-poster"></div>`;
      return `
        <button class="movie-result-card" type="button" data-index="${i}">
          ${poster}
          <div class="movie-result-info">
            <strong>${escapeHtml(m.title)}</strong>
            <span>${escapeHtml(m.year)}</span>
          </div>
        </button>`;
    })
    .join("");

  els.movieResults.querySelectorAll(".movie-result-card").forEach((btn, i) => {
    btn.addEventListener("click", () => selectMovie(movies[i]));
  });
}

async function selectMovie(movie) {
  clearError();
  els.searchStatus.textContent = `Fetching reviews for ${movie.title}…`;

  if (!analyzer) {
    showError("The model is still loading, one moment, then try again.");
    return;
  }

  try {
    const texts = await fetchReviews(movie.id, DEFAULT_API_KEY);
    if (!texts.length) {
      els.searchStatus.textContent = `TMDB has no written reviews for ${movie.title} (${movie.year}). Its review coverage is thin for a lot of titles; try the Paste tab with reviews copied from elsewhere instead.`;
      return;
    }
    const report = analyzer.analyzeTexts(texts);
    els.searchStatus.textContent = `Analyzed ${texts.length} review${texts.length === 1 ? "" : "s"} for ${movie.title} (${movie.year}) from TMDB.`;
    render(report);
  } catch (err) {
    showError(err.message || "Couldn't fetch reviews for that title.");
  }
}

els.movieSearchBtn.addEventListener("click", searchMovie);
els.movieQuery.addEventListener("keydown", (e) => {
  if (e.key === "Enter") searchMovie();
});

els.textarea.addEventListener("input", countReviews);
els.analyzeBtn.addEventListener("click", analyze);
els.sampleBtn.addEventListener("click", () => {
  els.textarea.value = SAMPLE_BATCH;
  countReviews();
});
els.showMoreBtn.addEventListener("click", () => {
  const collapsed = els.allList.classList.toggle("collapsed");
  els.showMoreBtn.textContent = collapsed ? "Show all reviews" : "Show fewer";
});

loadModel();

// Theme toggle: cycles light -> dark -> system, persisted locally.
function applyTheme(theme) {
  if (theme === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", theme);
  }
  els.themeToggle.textContent = theme === "dark" ? "Light mode" : theme === "light" ? "System theme" : "Dark mode";
}

let storedTheme = "system";
try {
  storedTheme = localStorage.getItem("audience-pulse-theme") || "system";
} catch (err) {
  // localStorage unavailable (private mode, etc), fall back to system theme silently
}
applyTheme(storedTheme);

els.themeToggle.addEventListener("click", () => {
  const next = storedTheme === "system" ? "dark" : storedTheme === "dark" ? "light" : "system";
  storedTheme = next;
  try {
    localStorage.setItem("audience-pulse-theme", next);
  } catch (err) {
    // ignore
  }
  applyTheme(next);
});
