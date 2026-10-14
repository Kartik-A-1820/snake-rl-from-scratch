// Interactive figures for tryout 02. Everything reads results.json written by python/train_dqn.py
// (+ results/bench/bench.json merged in as R.bench). Missing data → a clear "run it first" message.
const C2 = ['#306230', '#c27c0e', '#3a6ea5', '#9b3d8f', '#b3261e'];
const need = (host, H, cond, msg) => { if (!cond) host.append(H.el('div', { class: 'mono' }, msg)); return !cond; };

WIDGETS.summary = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Run summary'));
  if (need(host, H, R.config, 'No results yet: run  .\\run.ps1 train02  then  .\\run.ps1 notes')) return;
  const c = R.config, b = R.best, last = R.checkpoints[R.checkpoints.length - 1];
  const rows = [
    ['model', `${R.name}: ${c.params.toLocaleString()} params, ${(c.model_bytes_fp32 / 1e6).toFixed(2)} MB fp32`],
    ['device', c.device],
    ['replay buffer', `${c.buffer.toLocaleString()} transitions, ${(c.replay_bytes / 1e6).toFixed(0)} MB`],
    ['training', `${(R.epochs.at(-1)?.env_steps / 1e6).toFixed(1)} M env steps, ${c.envs} envs, batch ${c.batch}, lr ${c.lr}, γ ${c.gamma}, ${c.no_double ? 'vanilla' : 'Double'} DQN${c.dueling ? ' + dueling' : ''}`],
    ['last checkpoint (test, 500 seeds)', last ? `mean ${last.test.mean.toFixed(2)}, median ${last.test.median}, max ${last.test.max}, wins ${last.test.wins}` : '–'],
    ['best by validation', b ? `epoch ${b.epoch}: test mean ${b.test.mean.toFixed(2)}, max ${b.test.max}, wins ${b.test.wins}` : '–'],
    ['baselines (same 500 seeds)', `random ${R.baselines12.random}, greedy heuristic ${R.baselines12.greedy_heuristic}, tabular Q ${R.baselines12.tabular_q}`],
    ['throughput', R.epochs.length ? `${Math.round(R.epochs.reduce((s, e) => s + e.steps_per_s, 0) / R.epochs.length).toLocaleString()} env steps/s (avg incl. learning)` + (R.epochs.at(-1).vram_mb ? `, peak VRAM ${R.epochs.at(-1).vram_mb} MB` : '') : '–'],
  ];
  const t = H.el('table'); rows.forEach(([k, v]) => t.append(H.el('tr', {}, H.el('th', {}, k), H.el('td', {}, v)))); host.append(t);
};

WIDGETS.curves = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Learning curve: training (ε-greedy) per epoch, greedy test at checkpoints'));
  if (need(host, H, R.epochs && R.epochs.length, 'No training log yet.')) return;
  const chart = H.el('div'); host.append(chart);
  const ep = R.epochs.map(e => e.epoch);
  const cps = new Map(R.checkpoints.map(c => [c.epoch, c]));
  H.lineChart(chart, {
    series: [
      { name: 'train mean (ε-greedy)', x: ep, y: ep.map((_, i) => +R.epochs[i].train_mean.toFixed(2)), color: C2[1], dash: '4 3' },
      { name: 'TEST mean (greedy)', x: ep, y: ep.map(e => cps.has(e) ? +cps.get(e).test.mean.toFixed(2) : null), color: C2[0],
        extra: i => { const c = cps.get(ep[i]); return c ? `median ${c.test.median}, max ${c.test.max}, wins ${c.test.wins}\ndeaths ${JSON.stringify(c.test.causes)}` : ''; } },
    ],
    xlabel: 'epoch (50k env steps)', ylabel: 'score',
    hlines: [{ y: R.baselines12.greedy_heuristic, name: 'heuristic / tabular ≈ 20', color: '#b3261e' }],
  });
};

WIDGETS.checkpoint_table = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Checkpoints: every 25 epochs, greedy on 500 held-out test seeds'));
  if (need(host, H, R.checkpoints && R.checkpoints.length, 'No checkpoints yet.')) return;
  const pct = (c, k) => Math.round(100 * ((c.test.causes || {})[k] || 0) / c.test.n) + '%';
  const box = H.el('div');
  box.innerHTML = '<table><tr><th>epoch</th><th>env steps</th><th>ε</th><th>train mean (max, wins)</th><th>TEST mean</th><th>median</th><th>p90</th><th>max</th><th>wins</th><th>Δ test</th><th>wall / self / timeout</th><th>val mean</th></tr>' +
    R.checkpoints.map(c => `<tr><td>${c.epoch}</td><td>${(c.env_steps / 1e6).toFixed(2)}M</td><td>${c.eps}</td><td>${c.train_mean.toFixed(2)} (${c.train_max}, ${c.train_wins})</td>` +
      `<td><b>${c.test.mean.toFixed(2)}</b></td><td>${c.test.median}</td><td>${c.test.p90}</td><td>${c.test.max}</td><td>${c.test.wins}</td>` +
      `<td>${c.delta_test_mean == null ? '–' : (c.delta_test_mean >= 0 ? '+' : '') + c.delta_test_mean.toFixed(2)}</td><td>${pct(c, 'wall')} / ${pct(c, 'self')} / ${pct(c, 'timeout')}</td><td>${c.val_mean.toFixed(2)}</td></tr>`).join('') +
    '</table><div class="mono">win = score 141 on 12×12. Best checkpoint is chosen on validation seeds, never on test.</div>';
  host.append(box);
};

WIDGETS.loss_q = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Health checks: TD loss and mean predicted Q per epoch'));
  if (need(host, H, R.epochs && R.epochs.length, 'No training log yet.')) return;
  const a = H.el('div'), b = H.el('div'); host.append(a, b);
  const ep = R.epochs.map(e => e.epoch);
  H.lineChart(a, { series: [{ name: 'Huber TD loss', x: ep, y: R.epochs.map(e => +e.loss.toFixed(5)), color: C2[2] }], xlabel: 'epoch', ylabel: 'loss', h: 200 });
  H.lineChart(b, { series: [{ name: 'mean Q(s,a) on batches', x: ep, y: R.epochs.map(e => +e.q_mean.toFixed(3)), color: C2[3] }], xlabel: 'epoch', ylabel: 'Q', h: 200 });
  host.append(H.el('div', { class: 'mono' }, 'With γ = 0.99 and +1 per food, a policy eating every ~k steps has V ≈ 1/(1 − γ^k) − 1 … sanity-check that Q stays in that range instead of drifting upward (overestimation) or exploding (divergence).'));
};

WIDGETS.overestimation = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Monte Carlo: bias of max vs the Double estimator (true values all equal to 0)'));
  let m = 3, sigma = 1, n = 20000;
  const out = H.el('div', { class: 'mono', style: 'white-space:pre' }), chart = H.el('div');
  const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const draw = () => {
    let single = 0, dbl = 0;
    const hs = [], hd = [];
    for (let t = 0; t < n; t++) {
      const qa = Array.from({ length: m }, () => sigma * gauss()), qb = Array.from({ length: m }, () => sigma * gauss());
      const mx = Math.max(...qa); single += mx;
      const sel = qa.indexOf(mx); dbl += qb[sel];
      if (t < 400) { hs.push(mx); hd.push(qb[sel]); }
    }
    out.textContent = `actions m = ${m}, noise σ = ${sigma}\nE[max_a Q(a)]          = ${(single / n).toFixed(3)}   ← DQN target bias (true max = 0)\nE[Q_B(argmax_a Q_A(a))] = ${(dbl / n).toFixed(3)}   ← Double estimator`;
    const xs = Array.from({ length: 21 }, (_, i) => -3 + i * 0.3);
    const hist = arr => xs.map(x => arr.filter(v => v >= x && v < x + 0.3).length);
    H.lineChart(chart, { series: [{ name: 'max (single)', x: xs.map(x => +x.toFixed(1)), y: hist(hs), color: C2[4] }, { name: 'double', x: xs.map(x => +x.toFixed(1)), y: hist(hd), color: C2[0] }],
      xlabel: 'estimated value (σ units)', ylabel: 'count (400 draws)', h: 200 });
  };
  host.append(H.el('div', { class: 'row' }, H.slider('actions m ', 2, 20, 1, m, v => { m = v; draw(); }), H.slider('noise σ ', 0.1, 3, 0.1, sigma, v => { sigma = v; draw(); }, v => v.toFixed(1))), out, chart);
  draw();
};

WIDGETS.bench = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'CPU vs CUDA on this machine (python/bench_device.py)'));
  if (need(host, H, R.bench, 'No benchmark yet: run  .\\run.ps1 bench  then  .\\run.ps1 notes')) return;
  const B = R.bench;
  host.append(H.el('div', { class: 'mono' }, B.machine.devices.join(' | ')));
  const tbl = (rows, cols, title) => {
    const d = H.el('div'); d.innerHTML = `<p><b>${title}</b></p><table><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>` +
      rows.map(r => `<tr>${cols.map(c => `<td>${r[c] ?? ''}</td>`).join('')}</tr>`).join('') + '</table>';
    host.append(d);
  };
  tbl(B.inference, ['model', 'params', 'device', 'threads', 'latency_us'], '1. Inference latency at batch 1 (median µs, lower is better)');
  const ts = H.el('div'); host.append(H.el('p', {}, H.el('b', {}, '2. Training step throughput (samples/s, higher is better)')), ts);
  const series = [];
  ['mlp 11→64→64→3', 'cnn 4x14x14'].forEach((m, i) => ['cpu', 'cuda'].forEach((d, j) => {
    const rows = B.train_step.filter(r => r.model === m && r.device === d);
    if (rows.length) series.push({ name: `${m.split(' ')[0]} ${d}`, x: rows.map(r => r.batch), y: rows.map(r => r.samples_per_s), color: C2[i * 2 + j], dash: d === 'cpu' ? '4 3' : '' });
  }));
  H.lineChart(ts, { series, xlabel: 'batch size', ylabel: 'samples / s' });
  tbl(B.env, ['env', 'device', 'n_envs', 'steps_per_s'], '3. Environment throughput (env steps/s, random actions)');
};

WIDGETS.replay = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Replays of the best checkpoint on test seeds (exact re-simulation)'));
  if (need(host, H, R.replays && R.replays.length, 'No replays yet.')) return;
  let k = 0, g, t = 0, timer = null, speed = 15;
  const cv = H.el('canvas'), info = H.el('div', { class: 'mono', style: 'white-space:pre' }), bars = H.el('div');
  const sel = H.el('select'); R.replays.forEach((r, i) => sel.append(H.el('option', { value: i }, `seed ${r.seed}: score ${r.score} (${r.cause})`)));
  const draw = () => {
    const r = R.replays[k], q = r.q[Math.min(t, r.q.length - 1)];
    H.board(cv, g, { cell: 22 });
    info.textContent = `seed ${r.seed}  step ${t}/${r.actions.length}  score ${g.score}  length ${g.body.length}${g.done ? '  → ' + g.cause.toUpperCase() : ''}`;
    const lo = Math.min(...q), hi = Math.max(...q);
    bars.innerHTML = ['straight', 'right', 'left'].map((n, i) => `<div class="mono">${n.padEnd(9)} <span style="display:inline-block;height:10px;width:${Math.round(10 + 150 * (q[i] - lo) / ((hi - lo) || 1))}px;background:${q[i] === hi ? '#306230' : '#bbb'}"></span> ${q[i].toFixed(3)}</div>`).join('');
  };
  const load = () => { stop(); k = +sel.value; t = 0; g = new S.SnakeGame({ width: R.replays[k].width, height: R.replays[k].height, seed: R.replays[k].seed }); draw(); };
  const tick = () => { const r = R.replays[k]; if (g.done || t >= r.actions.length) return stop(); g.step(+r.actions[t++]); draw(); };
  const stop = () => { clearInterval(timer); timer = null; };
  sel.onchange = load;
  host.append(H.el('div', { class: 'row' }, sel, H.el('button', { class: 'btn', onclick: () => { stop(); timer = setInterval(tick, 1000 / speed); } }, '▶ play'),
    H.el('button', { class: 'btn', onclick: stop }, '❚❚'), H.el('button', { class: 'btn', onclick: () => { stop(); tick(); } }, 'step'),
    H.slider('speed ', 2, 60, 1, speed, v => { speed = v; if (timer) { stop(); timer = setInterval(tick, 1000 / speed); } }, v => v + '/s')),
    H.el('div', { class: 'row', style: 'align-items:flex-start' }, cv, H.el('div', {}, info, bars)));
  load();
};
