// Confirms the browser JS inference path (app/static/model.js + text.js)
// matches the Python NumPy path (app/model.py) that Keras itself has already
// been verified against (tests/test_model_matches_keras.py). This is the
// last link in the chain: Keras <-> NumPy <-> JavaScript should all agree,
// so the model the visitor's browser actually runs is provably the same
// model that was trained.
//
// Run after training/export_web_weights.py (which writes fixtures.json):
//
//   node scripts/verify_js_model.mjs

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { SentimentModel } from "../app/static/model.js";
import { Encoder } from "../app/static/text.js";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const WEB_MODEL_DIR = path.join(ROOT, "app", "static", "model");
const TOLERANCE = 1e-4;

async function loadFloat32(name) {
  const buf = await readFile(path.join(WEB_MODEL_DIR, name));
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

async function main() {
  const manifest = JSON.parse(await readFile(path.join(WEB_MODEL_DIR, "manifest.json"), "utf-8"));
  const vocabPayload = JSON.parse(await readFile(path.join(WEB_MODEL_DIR, "vocab.json"), "utf-8"));
  const fixtures = JSON.parse(await readFile(path.join(WEB_MODEL_DIR, "fixtures.json"), "utf-8"));

  const [embedding, rnnKernel, rnnRecurrent, rnnBias, denseW, denseB] = await Promise.all(
    ["embedding.bin", "rnn_kernel.bin", "rnn_recurrent.bin", "rnn_bias.bin", "dense_w.bin", "dense_b.bin"].map(loadFloat32)
  );

  const model = new SentimentModel({
    embedding,
    rnnKernel,
    rnnRecurrent,
    rnnBias,
    denseW,
    denseB,
    embeddingDim: manifest.embedding_dim,
    units: manifest.units,
  });
  const encoder = new Encoder(vocabPayload.vocab, vocabPayload.max_len);

  let failures = 0;
  for (const fixture of fixtures) {
    const tokenIds = encoder.encode(fixture.text);
    const jsScore = model.predict(tokenIds);
    const diff = Math.abs(jsScore - fixture.score);
    const ok = diff <= TOLERANCE;
    if (!ok) failures++;
    const status = ok ? "OK  " : "FAIL";
    console.log(`[${status}] js=${jsScore.toFixed(6)} py=${fixture.score.toFixed(6)} diff=${diff.toExponential(2)}  "${fixture.text.slice(0, 60)}..."`);
  }

  if (failures > 0) {
    console.error(`\n${failures}/${fixtures.length} fixtures mismatched (tolerance ${TOLERANCE}).`);
    process.exit(1);
  }
  console.log(`\nAll ${fixtures.length} fixtures match within ${TOLERANCE}. JS inference matches the Python model.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
