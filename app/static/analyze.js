// Batch review analysis, ported from app/analyze.py. Runs client-side, on
// top of SentimentModel (model.js) and Encoder (text.js).

export const POSITIVE_THRESHOLD = 0.5;
export const LOW_CONFIDENCE_COVERAGE = 0.6;
export const MAX_REVIEWS_PER_REQUEST = 200;

// Split pasted text into individual reviews. Reviews are expected one per
// blank-line-separated block (how people naturally paste a batch copied from
// a spreadsheet or review site). If the user pasted one review per single
// line instead, blank-line splitting yields one giant block, so fall back to
// splitting on single newlines.
export function splitReviews(rawText) {
  const blocks = rawText
    .split("\n\n")
    .map((b) => b.trim())
    .filter(Boolean);
  if (blocks.length > 1) return blocks;

  const lines = rawText
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length) return lines;

  const trimmed = rawText.trim();
  return trimmed ? [trimmed] : [];
}

export class Analyzer {
  constructor(model, encoder) {
    this.model = model;
    this.encoder = encoder;
  }

  analyzeOne(text) {
    const tokenIds = this.encoder.encode(text);
    const score = this.model.predict(tokenIds);
    const coverage = this.encoder.coverage(text);
    return {
      text,
      score,
      sentiment: score > POSITIVE_THRESHOLD ? "positive" : "negative",
      coverage,
      low_confidence: coverage < LOW_CONFIDENCE_COVERAGE,
    };
  }

  analyzeBatch(rawText) {
    const texts = splitReviews(rawText).slice(0, MAX_REVIEWS_PER_REQUEST);
    const reviews = texts.map((text) => this.analyzeOne(text));

    const positiveCount = reviews.filter((r) => r.sentiment === "positive").length;
    const negativeCount = reviews.length - positiveCount;
    const averageScore = reviews.length ? reviews.reduce((sum, r) => sum + r.score, 0) / reviews.length : 0;
    const pulseScore = reviews.length ? (positiveCount / reviews.length) * 100 : 0;
    const harshest = [...reviews].sort((a, b) => a.score - b.score).slice(0, 5);

    return {
      reviews,
      pulse_score: Math.round(pulseScore * 10) / 10,
      average_score: Math.round(averageScore * 10000) / 10000,
      positive_count: positiveCount,
      negative_count: negativeCount,
      harshest,
    };
  }
}
