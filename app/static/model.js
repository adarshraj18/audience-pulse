// Client-side inference, ported line-for-line from app/model.py.
//
// The whole point of this file: the model runs in the visitor's browser, not
// on a server. Training happens once, offline, in Keras (training/train.py).
// The trained weights are exported to raw Float32 binaries
// (training/export_web_weights.py) and this class replays the exact same
// forward pass (embedding lookup, a SimpleRNN scan, a sigmoid output) using
// nothing but arithmetic on those arrays. No ML framework ships to the
// visitor at all.
//
// scripts/verify_js_model.mjs checks this against app/model.py's fixtures.json
// so the two implementations can't quietly drift apart.

export class SentimentModel {
  constructor({ embedding, rnnKernel, rnnRecurrent, rnnBias, denseW, denseB, embeddingDim, units }) {
    this.embedding = embedding; // Float32Array, flattened (vocabSize * embeddingDim)
    this.rnnKernel = rnnKernel; // Float32Array, flattened (embeddingDim * units)
    this.rnnRecurrent = rnnRecurrent; // Float32Array, flattened (units * units)
    this.rnnBias = rnnBias; // Float32Array (units)
    this.denseW = denseW; // Float32Array (units)
    this.denseB = denseB; // Float32Array (1)
    this.embeddingDim = embeddingDim;
    this.units = units;
  }

  static async load(baseUrl) {
    const manifest = await (await fetch(`${baseUrl}/manifest.json`)).json();
    const [embedding, rnnKernel, rnnRecurrent, rnnBias, denseW, denseB] = await Promise.all(
      ["embedding.bin", "rnn_kernel.bin", "rnn_recurrent.bin", "rnn_bias.bin", "dense_w.bin", "dense_b.bin"].map(
        async (name) => new Float32Array(await (await fetch(`${baseUrl}/${name}`)).arrayBuffer())
      )
    );
    return new SentimentModel({
      embedding,
      rnnKernel,
      rnnRecurrent,
      rnnBias,
      denseW,
      denseB,
      embeddingDim: manifest.embedding_dim,
      units: manifest.units,
    });
  }

  // Run one padded/truncated token sequence through the network. Returns a
  // probability in [0, 1]: how positive the model reads the review.
  predict(tokenIds) {
    const { embeddingDim, units, embedding, rnnKernel, rnnRecurrent, rnnBias } = this;
    let h = new Float32Array(units);
    const preActivation = new Float32Array(units);

    for (const id of tokenIds) {
      const xOffset = id * embeddingDim;
      for (let u = 0; u < units; u++) {
        let sum = rnnBias[u];
        for (let e = 0; e < embeddingDim; e++) {
          sum += embedding[xOffset + e] * rnnKernel[e * units + u];
        }
        for (let v = 0; v < units; v++) {
          sum += h[v] * rnnRecurrent[v * units + u];
        }
        preActivation[u] = sum;
      }
      // Keras' SimpleRNN: h_t = tanh(x_t . W_kernel + h_{t-1} . W_recurrent + b)
      const nextH = new Float32Array(units);
      for (let u = 0; u < units; u++) nextH[u] = Math.tanh(preActivation[u]);
      h = nextH;
    }

    let logit = this.denseB[0];
    for (let u = 0; u < units; u++) logit += h[u] * this.denseW[u];
    return 1 / (1 + Math.exp(-logit));
  }
}
