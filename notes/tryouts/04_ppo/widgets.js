// Interactive figures for tryout 04. R = results.json written by python/make_notes_data.py:
//   R.name/config/epochs/checkpoints/best : the best PPO run (same shape as tryout 02's results.json plus entropy/kl/clipfrac)
//   R.arms     : [{id, label, runs: [{seed, run, best: {epoch, val_mean, test}, curve: [[epoch, testMean], ...]}]}]
//   R.parallel : {ppo: {rows: [...]}, dqn: {rows: [...]}}  from python/bench_parallel.py
const C4 = ['#306230', '#c27c0e', '#3a6ea5', '#9b3d8f', '#b3261e', '#5b6b2f'];
const need4 = (host, H, cond, msg) => { if (!cond) host.append(H.el('div', { class: 'mono' }, msg)); return !cond; };
const mean4 = a => a.reduce((s, x) => s + x, 0) / a.length;
const sd4 = a => { const m = mean4(a); return a.length < 2 ? 0 : Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };  // sample sd (n-1)

WIDGETS.summary = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Best PPO run'));
  if (need4(host, H, R.config, 'No results yet: run  python python/jobq.py  then  python python/make_notes_data.py')) return;
  const c = R.config, b = R.best, last = R.checkpoints[R.checkpoints.length - 1];
  const rows = [
    ['model', `${R.name}: ${c.params.toLocaleString()} params (shared CNN trunk, policy and value heads)`],
    ['device', c.device],
    ['setup', `${c.envs} envs × ${c.horizon} steps = ${c.batch.toLocaleString()} per update, ${c.update_epochs} epochs × ${c.minibatches} minibatches, lr ${c.lr}, γ ${c.gamma}, λ ${c.gae_lambda}, clip ${c.clip}, entropy ${c.ent_coef}${c.mask ? ', invalid-action masking' : ''}`],
    ['training', `${(R.epochs.at(-1).env_steps / 1e6).toFixed(1)} M env steps in ${(R.epochs.reduce((s, e) => s + e.seconds, 0) / 60).toFixed(0)} min (${Math.round(R.epochs.reduce((s, e) => s + e.steps_per_s, 0) / R.epochs.length).toLocaleString()} steps/s average)`],
    ['last checkpoint (test, 500 seeds)', last ? `mean ${last.test.mean.toFixed(2)}, median ${last.test.median}, max ${last.test.max}` : '–'],
    ['best by validation', b ? `epoch ${b.epoch}: test mean ${b.test.mean.toFixed(2)}, max ${b.test.max}, deaths ${JSON.stringify(b.test.causes)}` : '–'],
    ['reference (same seeds)', 'tabular ≈ 20, DQN CNN best 24.5'],
  ];
  const t = H.el('table'); rows.forEach(([k, v]) => t.append(H.el('tr', {}, H.el('th', {}, k), H.el('td', {}, v)))); host.append(t);
};

// GAE: advantage = sum_l (γλ)^l δ_{t+l}. Show the weights on the TD residuals and the effective horizon.
WIDGETS.gae_demo = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'GAE: how much each future TD residual δ counts'));
  let lam = 0.95;
  const gamma = 0.99, cv = H.el('canvas', { width: 720, height: 200 }), out = H.el('div', { class: 'mono', style: 'white-space:pre' });
  const draw = () => {
    const L = 40, ctx = cv.getContext('2d'); ctx.clearRect(0, 0, 720, 200);
    const w = Array.from({ length: L }, (_, l) => Math.pow(gamma * lam, l)), bw = 720 / L;
    w.forEach((v, l) => { ctx.fillStyle = '#306230'; ctx.fillRect(l * bw + 2, 180 - v * 160, bw - 4, v * 160); });
    ctx.fillStyle = '#333'; ctx.font = '10px monospace'; ctx.fillText('weight (γλ)^l on δ_{t+l}, l = 0 … 39', 6, 12);
    const hor = 1 / (1 - gamma * lam);
    out.textContent = `γ = ${gamma}, λ = ${lam.toFixed(2)}: weights decay by ${(gamma * lam).toFixed(3)} per step, effective horizon 1/(1-γλ) ≈ ${hor.toFixed(1)} steps.\n` +
      (lam === 0 ? 'λ = 0: one-step TD residual. Lowest variance, biased by the critic\'s errors.' :
        lam === 1 ? 'λ = 1: Monte-Carlo advantage G_t - V(s_t). Unbiased, highest variance.' : 'In between: the usual trade of a little bias for much less variance.');
  };
  host.append(H.slider('λ ', 0, 1, 0.01, lam, v => { lam = v; draw(); }, v => v.toFixed(2)), cv, out);
  draw();
};

// PPO clipped surrogate: L(r) = min(r A, clip(r, 1-ε, 1+ε) A) for A > 0 and A < 0.
WIDGETS.clip_demo = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'PPO clipped objective as a function of the ratio r = π_new / π_old'));
  let eps = 0.2;
  const cv = H.el('canvas', { width: 720, height: 260 }), out = H.el('div', { class: 'mono', style: 'white-space:pre' });
  const draw = () => {
    const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, 720, 260);
    const x = r => 30 + (r - 0.4) / 1.2 * 660, f = (r, A) => Math.min(r * A, Math.max(1 - eps, Math.min(1 + eps, r)) * A);
    const y = v => 130 - v * 70;
    ctx.strokeStyle = '#999'; ctx.beginPath(); ctx.moveTo(30, 130); ctx.lineTo(690, 130); ctx.moveTo(x(1), 10); ctx.lineTo(x(1), 250); ctx.stroke();
    [[1, '#306230', 'A > 0 (good action)'], [-1, '#b3261e', 'A < 0 (bad action)']].forEach(([A, col, lab], k) => {
      ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= 120; i++) { const r = 0.4 + i / 100, px = x(r), py = y(f(r, A)); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke(); ctx.fillStyle = col; ctx.font = '12px monospace'; ctx.fillText(lab, 40, 24 + 16 * k);
    });
    ctx.fillStyle = '#333'; ctx.font = '11px monospace';
    [0.5, 0.8, 1, 1.2, 1.5].forEach(r => ctx.fillText(r.toFixed(1), x(r) - 8, 256));
    ctx.fillText('r →', 672, 124);
    ctx.setLineDash([4, 4]); ctx.strokeStyle = '#c27c0e'; ctx.lineWidth = 1;
    [1 - eps, 1 + eps].forEach(r => { ctx.beginPath(); ctx.moveTo(x(r), 10); ctx.lineTo(x(r), 250); ctx.stroke(); });
    ctx.setLineDash([]);
    out.textContent = `ε = ${eps.toFixed(2)}. A > 0: the objective is flat for r > ${(1 + eps).toFixed(2)} (no gradient: the action is already ${(100 * eps).toFixed(0)}% more likely than before).\n` +
      `A < 0: flat for r < ${(1 - eps).toFixed(2)}. The min keeps the unclipped (worse) branch on the other side, so a step in the wrong direction is still corrected.`;
  };
  host.append(H.slider('ε ', 0.05, 0.5, 0.01, eps, v => { eps = v; draw(); }, v => v.toFixed(2)), cv, out);
  draw();
};

WIDGETS.parallel = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Measured: k identical jobs sharing the GTX 1650 Ti (python/bench_parallel.py)'));
  if (need4(host, H, R.parallel && (R.parallel.ppo || R.parallel.dqn), 'No benchmark yet: run  python python/bench_parallel.py --family ppo')) return;
  const chart = H.el('div');
  [['ppo', 'PPO (256 envs, update-heavy)'], ['dqn', 'DQN (32 envs, 4 updates per step)']].forEach(([fam, title], k) => {
    const P = R.parallel[fam]; if (!P) return;
    const t = H.el('table'); host.append(H.el('p', {}, H.el('b', {}, title)));
    t.append(H.el('tr', {}, ...['jobs', 'steps/s per job', 'aggregate steps/s', 'speed-up vs 1 job', 'GPU util %', 'CPU util %', 'GPU memory in use (MB)'].map(x => H.el('th', {}, x))));
    const base = P.rows[0].aggregate_steps_per_s;
    P.rows.forEach(r => t.append(H.el('tr', {}, H.el('td', {}, String(r.jobs)), H.el('td', {}, r.per_job_steps_per_s.map(v => v.toLocaleString()).join(' / ')),
      H.el('td', {}, r.aggregate_steps_per_s.toLocaleString()), H.el('td', {}, (r.aggregate_steps_per_s / base).toFixed(2) + '×'),
      H.el('td', {}, String(r.gpu_util_pct)), H.el('td', {}, String(r.cpu_util_pct)), H.el('td', {}, String(r.gpu_mem_used_mb)))));
    host.append(t);
  });
  host.append(chart);
  const series = ['ppo', 'dqn'].filter(f => R.parallel[f]).map((f, i) => {
    const rows = R.parallel[f].rows, base = rows[0].aggregate_steps_per_s;
    return { name: f.toUpperCase() + ' aggregate speed-up', x: rows.map(r => r.jobs), y: rows.map(r => +(r.aggregate_steps_per_s / base).toFixed(2)), color: C4[i * 2] };
  });
  H.lineChart(chart, { series, xlabel: 'concurrent jobs', ylabel: 'aggregate throughput vs 1 job', ymin: 0 });
};

WIDGETS.ppo_table = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'PPO arms: 500-game greedy test mean of the best checkpoint, per training seed'));
  if (need4(host, H, R.arms && R.arms.length, 'No arm data yet.')) return;
  const t = H.el('table');
  t.append(H.el('tr', {}, ...['arm', 'seeds', 'test mean per seed', 'mean ± sd', 'max (best seed)', 'deaths wall / self / timeout (mean per 500)'].map(x => H.el('th', {}, x))));
  R.arms.forEach(a => {
    const m = a.runs.map(r => r.best.test.mean), mx = a.runs.map(r => r.best.test.max);
    const cs = ['wall', 'self', 'timeout'].map(k => Math.round(mean4(a.runs.map(r => r.best.test.causes[k] || 0))));
    t.append(H.el('tr', {}, H.el('td', {}, a.label), H.el('td', {}, String(a.runs.length)), H.el('td', {}, m.map(x => x.toFixed(1)).join(', ')),
      H.el('td', {}, `${mean4(m).toFixed(1)} ± ${sd4(m).toFixed(1)}`), H.el('td', {}, String(Math.max(...mx))), H.el('td', {}, cs.join(' / '))));
  });
  host.append(t);
};

WIDGETS.ppo_curves = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Learning curves: greedy test mean (500 games) at each checkpoint, mean over seeds'));
  if (need4(host, H, R.arms && R.arms.length, 'No arm data yet.')) return;
  const chart = H.el('div'); host.append(chart);
  const series = R.arms.map((a, i) => {
    const eps = a.runs[0].curve.map(c => c[0]);
    return { name: a.label, x: eps.map(e => e * 0.5), y: eps.map((e, k) => +mean4(a.runs.filter(r => r.curve[k]).map(r => r.curve[k][1])).toFixed(2)), color: C4[i % C4.length] };
  });
  H.lineChart(chart, { series, xlabel: 'env steps (millions)', ylabel: 'test mean score', hlines: [{ y: 20, name: 'tabular ≈ 20', color: '#b3261e' }] });
  if (R.epochs && R.epochs.length) {
    const c2 = H.el('div'); host.append(H.el('h4', {}, `Training diagnostics of ${R.name}`), c2);
    const ep = R.epochs.map(e => e.epoch * 0.5);
    H.lineChart(c2, { series: [
      { name: 'policy entropy', x: ep, y: R.epochs.map(e => +e.entropy.toFixed(3)), color: C4[2] },
      { name: 'clip fraction ×10', x: ep, y: R.epochs.map(e => +(10 * e.clipfrac).toFixed(3)), color: C4[1] },
      { name: 'approx KL ×100', x: ep, y: R.epochs.map(e => +(100 * e.approx_kl).toFixed(3)), color: C4[3] },
    ], xlabel: 'env steps (millions)', ylabel: 'value', h: 220 });
  }
};
