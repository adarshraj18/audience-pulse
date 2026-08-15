// Text encoding, ported line-for-line from app/text.py.
//
// The rules mirror keras.datasets.imdb: every sequence starts with id 1,
// unknown words map to id 2, and real word ids are shifted by 3 so those two
// reserved slots plus 0 (padding) don't collide with vocabulary entries.
// This has to match the Python encoder exactly, since it was used to
// generate the vocabulary the model was trained on.

const START_ID = 1;
const OOV_ID = 2;

const HTML_BREAK_RE = /<br\s*\/?>/gi;
const TOKEN_RE = /[a-z0-9']+/g;

export function tokenize(text) {
  const cleaned = text.replace(HTML_BREAK_RE, " ").toLowerCase();
  return cleaned.match(TOKEN_RE) || [];
}

export class Encoder {
  constructor(vocab, maxLen) {
    this.vocab = vocab; // plain object: word -> id
    this.maxLen = maxLen;
  }

  static async load(url) {
    const res = await fetch(url);
    const payload = await res.json();
    return new Encoder(payload.vocab, payload.max_len);
  }

  // Encode text, then pre-pad or pre-truncate to maxLen. Truncating from the
  // front matches training (pad_sequences default truncating='pre'): a
  // review's verdict usually lands in its closing sentences, so if something
  // has to be dropped it's the opening, not the ending.
  encode(text) {
    const words = tokenize(text);
    const ids = [START_ID, ...words.map((w) => (Object.prototype.hasOwnProperty.call(this.vocab, w) ? this.vocab[w] : OOV_ID))];
    if (ids.length >= this.maxLen) {
      return ids.slice(ids.length - this.maxLen);
    }
    return new Array(this.maxLen - ids.length).fill(0).concat(ids);
  }

  // Share of words the model actually recognises, in [0, 1]. Surfaced in the
  // UI as an honesty signal alongside the sentiment score.
  coverage(text) {
    const words = tokenize(text);
    if (words.length === 0) return 0;
    const known = words.filter((w) => Object.prototype.hasOwnProperty.call(this.vocab, w)).length;
    return known / words.length;
  }
}
