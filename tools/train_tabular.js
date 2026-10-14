// Tabular Q-learning on the 11 basic features, with a checkpoint report every N epochs.
// Run:  node tools/train_tabular.js            (12x12, writes results/ and models/)
//       node tools/train_tabular.js --size 6   (small board where a win is actually possible)
// Deterministic: same args → same numbers, because the engine and the RNG are seeded.
// No PyTorch here on purpose: a 2048x3 table needs no GPU, and pretending otherwise would be theatre.
const fs = require('fs');
const path = require('path');
const S = require('../game/snake_core.js');

function runTabular(S, cfg) {
  const c = Object.assign({
    size: 12, epochs: 200, episodesPerEpoch: 100, checkpointEvery: 25,
    alpha: 0.1, gamma: 0.95, epsStart: 1.0, epsEnd: 0.01, epsDecayFrac: 0.5,
    seed: 1, evalEpisodes: 500, evalSeed0: 100000,
  }, cfg);
  const totalEps = c.epochs * c.episodesPerEpoch;
  const winScore = c.size * c.size - 3; // board full = length size², start length 3
  const q = Array.from({ length: 2048 }, () => [0, 0, 0]);
  const visits = new Uint32Array(2048);
  const rng = new S.Rng(c.seed ^ 0x9E3779B9);
  const g = new S.SnakeGame({ width: c.size, height: c.size, seed: c.seed });
  // Eval seeds are disjoint from training (training uses one continuing stream from c.seed).
  const evalSeeds = Array.from({ length: c.evalEpisodes }, (_, i) => c.evalSeed0 + i);
  const greedy = (gm) => S.argmax(q[S.basicStateIndex(gm)]);

  const checkpoints = [];
  let win = { scores: [], causes: {}, wins: 0, steps: 0 };
  let totalSteps = 0;
  const t0 = Date.now();

  const checkpoint = (epoch, eps) => {
    const ev = S.evaluate(greedy, { width: c.size, height: c.size, seeds: evalSeeds });
    const tr = win.scores.length
      ? { episodes: win.scores.length, mean: win.scores.reduce((a, b) => a + b, 0) / win.scores.length,
          max: Math.max(...win.scores), wins: win.wins, causes: win.causes }
      : null;
    const prev = checkpoints[checkpoints.length - 1];
    const cp = {
      epoch, episodes: epoch * c.episodesPerEpoch, envSteps: totalSteps, eps: +eps.toFixed(3),
      statesVisited: Array.from(visits).filter(v => v > 0).length,
      train: tr, // scores while exploring (epsilon-greedy) during the last window
      eval: { ...ev.summary, wins: ev.causes.win || 0, causes: ev.causes }, // greedy, held-out seeds
      deltaEvalMean: prev ? ev.summary.mean - prev.eval.mean : null,
      bestFillPct: +(100 * (ev.summary.max + 3) / (c.size * c.size)).toFixed(1),
      wallSeconds: (Date.now() - t0) / 1000,
    };
    checkpoints.push(cp);
    win = { scores: [], causes: {}, wins: 0, steps: 0 };
    return cp;
  };

  checkpoint(0, c.epsStart);
  for (let ep = 0; ep < totalEps; ep++) {
    const eps = Math.max(c.epsEnd, c.epsStart - (c.epsStart - c.epsEnd) * ep / (totalEps * c.epsDecayFrac));
    g.reset();
    let s = S.basicStateIndex(g);
    while (!g.done) {
      const a = rng.nextU32() / 4294967296 < eps ? rng.int(3) : S.argmax(q[s]);
      const r = g.step(a);
      const s2 = r.terminated ? -1 : S.basicStateIndex(g);
      // terminated → no bootstrap; truncated (timeout) → bootstrap, the state wasn't terminal
      const target = r.reward + (r.terminated ? 0 : c.gamma * Math.max(...q[s2]));
      q[s][a] += c.alpha * (target - q[s][a]);
      visits[s]++; s = s2; totalSteps++;
    }
    win.scores.push(g.score);
    win.causes[g.cause] = (win.causes[g.cause] || 0) + 1;
    if (g.won) win.wins++;
    const epoch = (ep + 1) / c.episodesPerEpoch;
    if (Number.isInteger(epoch) && epoch % c.checkpointEvery === 0) checkpoint(epoch, eps);
  }

  const sparse = {};
  q.forEach((row, i) => { if (visits[i]) sparse[i] = row.map(v => +v.toFixed(4)); });
  return { config: c, winScore, checkpoints, model: { type: 'tabular', features: 'basic', size: c.size, q: sparse } };
}

function table(res) {
  const L = [`board ${res.config.size}x${res.config.size}  (win = score ${res.winScore})  epoch = ${res.config.episodesPerEpoch} episodes`,
    'epoch | eps   | train mean (max, wins) | EVAL mean  median p90 max | wins | Δ eval mean | deaths wall/self/timeout'];
  for (const cp of res.checkpoints) {
    const e = cp.eval, t = cp.train;
    const tr = t ? `${t.mean.toFixed(2).padStart(6)} (${String(t.max).padStart(3)}, ${t.wins})` : '     –           ';
    const d = cp.deltaEvalMean === null ? '     –' : (cp.deltaEvalMean >= 0 ? '+' : '') + cp.deltaEvalMean.toFixed(2);
    const n = e.n;
    const pc = k => ((100 * (e.causes[k] || 0)) / n).toFixed(0) + '%';
    L.push(`${String(cp.epoch).padStart(5)} | ${cp.eps.toFixed(3)} | ${tr.padEnd(22)} | ${e.mean.toFixed(2).padStart(6)} ${String(e.median).padStart(6)} ${String(e.p90).padStart(3)} ${String(e.max).padStart(3)} | ${String(e.wins).padStart(4)} | ${d.padStart(11)} | ${pc('wall')}/${pc('self')}/${pc('timeout')}`);
  }
  return L.join('\n');
}

module.exports = { runTabular, table };

if (require.main === module) {
  const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? +process.argv[i + 1] : d; };
  const res = runTabular(S, { size: arg('size', 12), epochs: arg('epochs', 200), seed: arg('seed', 1) });
  const root = path.join(__dirname, '..');
  const tag = `${res.config.size}x${res.config.size}_seed${res.config.seed}`;
  fs.mkdirSync(path.join(root, 'results', 'phase1_tabular'), { recursive: true });
  fs.mkdirSync(path.join(root, 'models'), { recursive: true });
  const { model, ...report } = res;
  fs.writeFileSync(path.join(root, 'results', 'phase1_tabular', `checkpoints_${tag}.json`), JSON.stringify(report, null, 1));
  fs.writeFileSync(path.join(root, 'models', `tabular_q_${tag}.json`), JSON.stringify(model));
  console.log(table(res));
}
