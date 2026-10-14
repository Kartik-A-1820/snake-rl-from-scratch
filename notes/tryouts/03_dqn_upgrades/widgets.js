// Interactive figures for tryout 03. R = results.json written by python/make_notes_data.py:
//   R.arms  : [{id, label, components, runs: [{seed, run, best: {epoch, val_mean, test}, curve: [[epoch, testMean], ...]}]}]
//   R.name/config/epochs/checkpoints/best : the long run of the full combination (same shape as tryout 02's results.json)
const C3 = ['#306230', '#c27c0e', '#3a6ea5', '#9b3d8f', '#b3261e', '#5b6b2f'];
const need3 = (host, H, cond, msg) => { if (!cond) host.append(H.el('div', { class: 'mono' }, msg)); return !cond; };
const mean3 = a => a.reduce((s, x) => s + x, 0) / a.length;
const sd3 = a => { const m = mean3(a); return a.length < 2 ? 0 : Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };  // sample sd (n-1)

WIDGETS.summary = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Long run: all four upgrades together'));
  if (need3(host, H, R.config, 'No long run yet: run  python python/plan_tonight.py && python python/jobq.py --plan results/plan_tonight.json  then  python python/make_notes_data.py')) return;
  const c = R.config, b = R.best, last = R.checkpoints[R.checkpoints.length - 1];
  const rows = [
    ['model', `${R.name}: ${c.params.toLocaleString()} params`],
    ['device', c.device],
    ['components', `Double DQN${c.dueling ? ' + dueling' : ''}${c.per ? ' + prioritized replay' : ''}${c.mask ? ' + action masking' : ''}, ${c.n_step}-step returns`],
    ['training', `${(R.epochs.at(-1).env_steps / 1e6).toFixed(1)} M env steps, ${c.envs} envs, batch ${c.batch}, ${(R.epochs.reduce((s, e) => s + e.seconds, 0) / 60).toFixed(0)} min`],
    ['last checkpoint (test, 500 seeds)', last ? `mean ${last.test.mean.toFixed(2)}, median ${last.test.median}, max ${last.test.max}` : '–'],
    ['best by validation', b ? `epoch ${b.epoch}: test mean ${b.test.mean.toFixed(2)}, max ${b.test.max}, deaths ${JSON.stringify(b.test.causes)}` : '–'],
    ['reference (same seeds)', 'tabular ≈ 20, tryout 02 CNN best 24.5 (last-8 mean 21.1)'],
  ];
  const t = H.el('table'); rows.forEach(([k, v]) => t.append(H.el('tr', {}, H.el('th', {}, k), H.el('td', {}, v)))); host.append(t);
};

WIDGETS.ablation_table = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Ablation: 1 M steps, test mean of the best checkpoint, per training seed'));
  if (need3(host, H, R.arms && R.arms.length, 'No ablation data yet.')) return;
  const t = H.el('table');
  t.append(H.el('tr', {}, ...['arm', 'seeds', 'test mean per seed', 'mean ± sd', 'max (best seed)', 'deaths wall / self / timeout (mean per 500)'].map(x => H.el('th', {}, x))));
  R.arms.forEach(a => {
    const m = a.runs.map(r => r.best.test.mean), mx = a.runs.map(r => r.best.test.max);
    const cs = ['wall', 'self', 'timeout'].map(k => Math.round(mean3(a.runs.map(r => r.best.test.causes[k] || 0))));
    t.append(H.el('tr', {}, H.el('td', {}, a.label), H.el('td', {}, String(a.runs.length)), H.el('td', {}, m.map(x => x.toFixed(1)).join(', ')),
      H.el('td', {}, `${mean3(m).toFixed(1)} ± ${sd3(m).toFixed(1)}`), H.el('td', {}, String(Math.max(...mx))), H.el('td', {}, cs.join(' / '))));
  });
  host.append(t);
};

WIDGETS.ablation_curves = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Greedy test mean during training, mean over seeds (500 held-out games per point)'));
  if (need3(host, H, R.arms && R.arms.length, 'No ablation data yet.')) return;
  const chart = H.el('div'); host.append(chart);
  const series = R.arms.map((a, i) => {
    const eps = a.runs[0].curve.map(c => c[0]);
    const ys = eps.map((e, k) => mean3(a.runs.filter(r => r.curve[k]).map(r => r.curve[k][1])));
    return { name: a.label, x: eps.map(e => e * 0.05), y: ys.map(v => +v.toFixed(2)), color: C3[i % C3.length] };
  });
  H.lineChart(chart, { series, xlabel: 'env steps (millions)', ylabel: 'test mean score', hlines: [{ y: 20, name: 'tabular ≈ 20', color: '#b3261e' }] });
};

WIDGETS.long_curves = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Long run (5 M steps): training score and greedy test score'));
  if (need3(host, H, R.epochs && R.epochs.length, 'No long run yet.')) return;
  const chart = H.el('div'); host.append(chart);
  const ep = R.epochs.map(e => e.epoch * 0.05);
  const cps = R.checkpoints;
  H.lineChart(chart, {
    series: [
      { name: 'train mean (ε-greedy)', x: ep, y: R.epochs.map(e => +e.train_mean.toFixed(2)), color: C3[1], dash: '4 3' },
      { name: 'TEST mean (greedy, 500 seeds)', x: cps.map(c => c.epoch * 0.05), y: cps.map(c => +c.test.mean.toFixed(2)), color: C3[0] },
    ], xlabel: 'env steps (millions)', ylabel: 'score', hlines: [{ y: 20, name: 'tabular ≈ 20', color: '#b3261e' }],
  });
};

// Prioritized replay: priorities p_i = (|δ_i| + ε)^α, P(i) ∝ p_i, weights w_i = (N P(i))^-β / max w.
WIDGETS.per_demo = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Prioritized replay: 12 transitions with TD errors from tiny to large'));
  const delta = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0];
  let alpha = 0.6, beta = 0.4;
  const out = H.el('div', { class: 'mono', style: 'white-space:pre' }), cv = H.el('canvas', { width: 720, height: 220 });
  const draw = () => {
    const N = delta.length, p = delta.map(d => Math.pow(d + 1e-6, alpha)), Z = p.reduce((a, b) => a + b, 0);
    const P = p.map(x => x / Z), wRaw = P.map(x => Math.pow(1 / (N * x), beta)), wMax = Math.max(...wRaw), w = wRaw.map(x => x / wMax);
    const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, 720, 220);
    const bw = 720 / N;
    delta.forEach((d, i) => {
      ctx.fillStyle = '#306230'; const h1 = P[i] * 190 * 2.2; ctx.fillRect(i * bw + 6, 200 - Math.min(190, h1), bw * 0.4, Math.min(190, h1));
      ctx.fillStyle = '#c27c0e'; const h2 = w[i] * 170; ctx.fillRect(i * bw + 6 + bw * 0.42, 200 - h2, bw * 0.4, h2);
      ctx.fillStyle = '#333'; ctx.font = '10px monospace'; ctx.fillText(d.toFixed(2), i * bw + 6, 214);
    });
    ctx.fillStyle = '#306230'; ctx.fillText('■ sampling probability P(i)', 8, 12); ctx.fillStyle = '#c27c0e'; ctx.fillText('■ importance weight w_i (max = 1)', 200, 12);
    const unif = 1 / N;
    out.textContent = `uniform P = ${unif.toFixed(3)} each.  α = ${alpha.toFixed(2)}: the largest error is sampled ${(P[N - 1] / unif).toFixed(1)}× as often as uniform, ` +
      `the smallest ${(P[0] / unif).toFixed(2)}×.\nβ = ${beta.toFixed(2)}: its update is scaled by w = ${w[N - 1].toFixed(2)} (smallest error: ${w[0].toFixed(2)}). β = 1 would make the expected update exactly the uniform one.`;
  };
  host.append(H.el('div', { class: 'row' }, H.slider('α (priority exponent) ', 0, 1, 0.05, alpha, v => { alpha = v; draw(); }, v => v.toFixed(2)),
    H.slider('β (importance-sampling) ', 0, 1, 0.05, beta, v => { beta = v; draw(); }, v => v.toFixed(2))), cv, out);
  draw();
};
