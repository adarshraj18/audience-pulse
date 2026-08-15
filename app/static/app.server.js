// Alternate frontend entry point: talks to the server's /api/analyze instead
// of scoring in the browser. NOT loaded by default; index.html loads app.js,
// which runs inference client-side (see app.js's header comment for why:
// free-tier server CPUs were too slow for this workload in practice).
//
// Kept for local development against the FastAPI backend, or for deploying
// on infrastructure with real CPU where server-side scoring makes sense: swap
// the <script> tag in index.html from app.js to this file.

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
};

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
  els.errorBanner.classList.remove("active");
  if (!text) {
    els.errorBanner.textContent = "Paste at least one review first.";
    els.errorBanner.classList.add("active");
    return;
  }

  els.analyzeBtn.disabled = true;
  els.spinner.classList.add("active");
  els.analyzeLabel.textContent = "Analyzing…";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "Something went wrong analyzing that batch.");
    }
    const report = await res.json();
    render(report);
  } catch (err) {
    els.errorBanner.textContent = err.message || "Couldn't reach the analysis service.";
    els.errorBanner.classList.add("active");
  } finally {
    els.analyzeBtn.disabled = false;
    els.spinner.classList.remove("active");
    els.analyzeLabel.textContent = "Analyze pulse";
  }
}

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
