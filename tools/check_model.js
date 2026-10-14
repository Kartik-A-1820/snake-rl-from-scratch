// Proves the browser's inference (runLayers in game/index.html) matches PyTorch on the test vectors that
// python/export_model.py stored inside each exported model. Usage: node tools/check_model.js [models/x.json ...]
const fs = require('fs');
const path = require('path');
const S = require('../game/snake_core.js');

const dir = path.join(__dirname, '..', 'models');
const files = process.argv.slice(2).length ? process.argv.slice(2)
  : [...fs.readdirSync(dir).filter(f => f.endsWith('.json')).map(f => path.join(dir, f)),
     ...fs.readdirSync(dir, { withFileTypes: true }).filter(d => d.isDirectory() && fs.existsSync(path.join(dir, d.name, 'best.json')))
       .map(d => path.join(dir, d.name, 'best.json'))];
let bad = 0, checked = 0;
for (const f of files) {
  const m = JSON.parse(fs.readFileSync(f, 'utf8'));
  if (m.type !== 'net' || !m.tests) continue;
  let worst = 0, flips = 0;
  const t0 = process.hrtime.bigint();
  for (const t of m.tests) {
    const q = S.runLayers(m.layers, { data: Float32Array.from(t.x), shape: t.shape }).data;
    q.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - t.q[i])); });
    if (S.argmax(Array.from(q)) !== S.argmax(t.q)) flips++;
  }
  const us = Number(process.hrtime.bigint() - t0) / 1e3 / m.tests.length;
  const ok = worst < 1e-3 && flips === 0;
  if (!ok) bad++;
  checked++;
  console.log(`${ok ? 'OK  ' : 'FAIL'} ${path.relative(dir, f)}: max |JS − torch| = ${worst.toExponential(2)}, argmax flips ${flips}/${m.tests.length}, ` +
    `JS inference ≈ ${us.toFixed(0)} µs/state (node, single thread)`);
}
if (!checked) console.log('no exported nets with test vectors found in models/');
process.exit(bad ? 1 : 0);
