// Writes reference trajectories from the JS engine (the source of truth for the rules).
// python/tests/test_parity.py replays the same actions in the Python envs and must match step for step.
// Usage: node tools/make_trace.js [outfile]
const fs = require('fs');
const path = require('path');
const S = require('../game/snake_core.js');

// Order-sensitive digest of the grid observation, so Python can compare without shipping whole tensors.
function obsDigest(g) {
  const { data } = S.gridObservation(g);
  let d = 0;
  for (let i = 0; i < data.length; i++) if (data[i]) d += data[i] * ((i % 97) + 1);
  return d;
}

function trace(size, seed, nSteps) {
  const g = new S.SnakeGame({ width: size, height: size, seed });
  const rng = new S.Rng(seed ^ 0xABCDEF);
  const steps = [];
  for (let t = 0; t < nSteps; t++) {
    // Mostly greedy so episodes get long (bodies, growth, self-collisions); some random for coverage.
    const a = rng.nextU32() / 4294967296 < 0.75 ? S.policies.greedy(g) : rng.int(3);
    const r = g.step(a);
    const bits = S.basicFeatures(g).reduce((s, b, i) => s | (b << i), 0);
    steps.push({
      a, ev: r.event, rew: r.reward, term: r.terminated, trunc: r.truncated,
      head: g.body[0], dir: g.dir, food: g.food, score: g.score, len: g.body.length,
      feat: bits, obs: +obsDigest(g).toFixed(6),
    });
    if (g.done) g.reset(); // continue the same RNG stream, like the vec env's auto-reset
  }
  return { size, seed, steps };
}

const cases = [];
for (const size of [4, 6, 12, 20]) for (const seed of [0, 1, 42, 12345]) cases.push(trace(size, seed, size <= 6 ? 3000 : 6000));
const rng = new S.Rng(42);
const out = { rng42: Array.from({ length: 8 }, () => rng.nextU32()), cases };
const file = process.argv[2] || path.join(__dirname, '..', 'python', 'tests', 'js_traces.json');
fs.writeFileSync(file, JSON.stringify(out));
console.log(`wrote ${cases.length} traces (${cases.reduce((n, c) => n + c.steps.length, 0)} steps) to ${file}`);
