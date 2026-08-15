// Minimal client for The Movie Database (TMDB) API, used by the "search a
// movie" mode. Free for non-commercial use; get a key at
// https://www.themoviedb.org/settings/api. The key is entered by each visitor
// and kept only in their browser's localStorage, never sent anywhere but
// TMDB itself, and never committed to this repo.
//
// TMDB's review coverage is real but thin (often a handful of reviews for a
// big release, sometimes none for a smaller one), nowhere near the volume a
// site like Rotten Tomatoes shows on its own pages, which has no public API.
// That's disclosed in the UI rather than papered over.

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
