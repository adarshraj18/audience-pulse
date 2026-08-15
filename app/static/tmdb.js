// Minimal client for The Movie Database (TMDB) API, used by the "search a
// movie" mode. Free for non-commercial use: https://www.themoviedb.org/settings/api.
//
// DEFAULT_API_KEY below is a real key so every visitor can search with zero
// setup, rather than each one needing their own account. TMDB v3 keys are
// designed to be used exactly like this, directly in client-side code with
// no server to hide them behind, so this isn't a leaked secret. The
// trade-off worth naming: it's one key covering all of this app's traffic,
// so TMDB's rate limit is shared rather than per-visitor. If that ever
// becomes a real problem, the fix is a visitor-provided key again, not
// anything more complicated.
//
// TMDB's review coverage is real but thin (often a handful of reviews for a
// big release, sometimes none for a smaller one), nowhere near the volume a
// site like Rotten Tomatoes shows on its own pages, which has no public API.
// That's disclosed in the UI rather than papered over.

export const DEFAULT_API_KEY = "661039062b68573fe66979497038fd88";

const BASE_URL = "https://api.themoviedb.org/3";
const POSTER_BASE = "https://image.tmdb.org/t/p/w92";

class TmdbError extends Error {}

async function request(path, apiKey) {
  const url = new URL(`${BASE_URL}${path}`);
  url.searchParams.set("api_key", apiKey);
  const res = await fetch(url.toString());
  if (res.status === 401) {
    throw new TmdbError("That API key was rejected by TMDB. Double-check it and try again.");
  }
  if (!res.ok) {
    throw new TmdbError(`TMDB request failed (HTTP ${res.status}). Try again in a moment.`);
  }
  return res.json();
}

export async function searchMovies(query, apiKey) {
  const data = await request(`/search/movie?query=${encodeURIComponent(query)}`, apiKey);
  return (data.results || [])
    .sort((a, b) => b.popularity - a.popularity)
    .slice(0, 6)
    .map((m) => ({
      id: m.id,
      title: m.title,
      year: (m.release_date || "").slice(0, 4) || "n/a",
      posterUrl: m.poster_path ? `${POSTER_BASE}${m.poster_path}` : null,
    }));
}

export async function fetchReviews(movieId, apiKey) {
  const data = await request(`/movie/${movieId}/reviews?page=1`, apiKey);
  return (data.results || []).map((r) => r.content.trim()).filter(Boolean);
}

export { TmdbError };
