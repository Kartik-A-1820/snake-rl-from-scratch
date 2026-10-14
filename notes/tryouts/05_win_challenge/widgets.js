// Interactive figures for tryout 05. R = results.json written by python/make_notes_data.py:
//   R.stages : [{size, label, runs: [{name, init_from, scratch, best: {epoch, test}, curve: [[epoch, mean, wins], ...], first_win, steps}]}]
//   R.hamilton : [{size, win_rate, mean, max}]      R.replays : [{label, seed, width, height, actions, score, cause}]
const C5 = ['#306230', '#c27c0e', '#3a6ea5', '#9b3d8f', '#b3261e'];
const need5 = (host, H, cond, msg) => { if (!cond) host.append(H.el('div', { class: 'mono' }, msg)); return !cond; };
const pct5 = (w, n) => (100 * w / n).toFixed(1) + '%';

WIDGETS.summary = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Where each board size stands'));
  if (need5(host, H, R.stages && R.stages.length, 'No runs yet: run  python python/plan_tonight.py  then  python python/jobq.py --plan results/plan_tonight.json')) return;
  const t = H.el('table');
  t.append(H.el('tr', {}, ...['board', 'run', 'test win rate (500 games)', 'mean fill', 'best score / cells', 'first win'].map(x => H.el('th', {}, x))));
  R.stages.forEach(st => st.runs.forEach(r => {
    const b = r.best, tt = b.test, cells = st.size * st.size - 3;
    t.append(H.el('tr', {}, H.el('td', {}, `${st.size}×${st.size}`), H.el('td', {}, r.name + (r.scratch ? ' (from scratch)' : '')),
      H.el('td', {}, pct5(tt.wins, tt.n || 500)), H.el('td', {}, (100 * tt.mean / cells).toFixed(0) + '%'),
      H.el('td', {}, `${tt.max} / ${cells}`), H.el('td', {}, r.first_win ? `${(r.first_win.env_steps / 1e6).toFixed(1)} M steps` : 'none yet')));
  }));
  host.append(t);
};

WIDGETS.horizon_demo = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Discounting: how much a reward T steps away is worth now'));
  let gamma = 0.995, T = 300, bonus = 10;
  const cv = H.el('canvas', { width: 720, height: 220 }), out = H.el('div', { class: 'mono', style: 'white-space:pre' });
  const draw = () => {
    const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, 720, 220);
    const L = 800, X = t => 30 + t / L * 680, Y = v => 200 - v * 180;
    ctx.strokeStyle = '#bbb'; ctx.beginPath(); ctx.moveTo(30, 200); ctx.lineTo(710, 200); ctx.moveTo(30, 20); ctx.lineTo(30, 200); ctx.stroke();
    [[0.99, '#999'], [0.995, '#c27c0e'], [0.999, '#3a6ea5']].forEach(([g, col]) => {
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath();
      for (let t = 0; t <= L; t += 4) { const x = X(t), y = Y(Math.pow(g, t)); t ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke();
      ctx.fillStyle = col; ctx.font = '11px monospace'; ctx.fillText('γ = ' + g, 620, Y(Math.pow(g, 760)) - 4);
    });
    ctx.strokeStyle = '#306230'; ctx.lineWidth = 3; ctx.beginPath();
    for (let t = 0; t <= L; t += 4) { const x = X(t), y = Y(Math.pow(gamma, t)); t ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke();
    ctx.fillStyle = '#b3261e'; ctx.beginPath(); ctx.arc(X(Math.min(T, L)), Y(Math.pow(gamma, T)), 5, 0, 7); ctx.fill();
    ctx.fillStyle = '#333'; ctx.font = '11px monospace'; ctx.fillText('weight γ^T  (T = steps until the reward, 0 … 800)', 36, 14);
    const w = Math.pow(gamma, T);
    out.textContent = `γ = ${gamma.toFixed(3)}: effective horizon 1/(1-γ) = ${(1 / (1 - gamma)).toFixed(0)} steps.\n` +
      `A bonus of ${bonus} arriving after T = ${T} steps is worth ${bonus} × ${w.toFixed(4)} = ${(bonus * w).toFixed(2)} at the decision that led to it.`;
  };
  host.append(H.el('div', { class: 'row' },
    H.slider('γ ', 0.95, 0.999, 0.001, gamma, v => { gamma = v; draw(); }, v => v.toFixed(3)),
    H.slider('T ', 10, 800, 10, T, v => { T = v; draw(); }, v => String(v)),
    H.slider('bonus ', 1, 50, 1, bonus, v => { bonus = v; draw(); }, v => String(v))), cv, out);
  draw();
};

WIDGETS.curriculum_table = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Curriculum results: win rate on 500 held-out seeds, best checkpoint per run'));
  if (need5(host, H, R.stages && R.stages.length, 'No runs yet.')) return;
  const t = H.el('table');
  t.append(H.el('tr', {}, ...['board', 'run', 'start from', 'win rate', 'mean score (fill %)', 'max', 'wins / games', 'deaths wall / self / timeout', 'steps'].map(x => H.el('th', {}, x))));
  R.stages.forEach(st => st.runs.forEach(r => {
    const tt = r.best.test, cells = st.size * st.size - 3, c = tt.causes || {};
    t.append(H.el('tr', {}, H.el('td', {}, `${st.size}×${st.size}`), H.el('td', {}, r.name), H.el('td', {}, r.init_from || 'scratch'),
      H.el('td', {}, `<b>${pct5(tt.wins, tt.n || 500)}</b>`.replace(/<[^>]+>/g, '')), H.el('td', {}, `${tt.mean.toFixed(1)} (${(100 * tt.mean / cells).toFixed(0)}%)`), H.el('td', {}, String(tt.max)),
      H.el('td', {}, `${tt.wins} / ${tt.n || 500}`), H.el('td', {}, `${c.wall || 0} / ${c.self || 0} / ${c.timeout || 0}`), H.el('td', {}, (r.steps / 1e6).toFixed(1) + ' M')));
  }));
  host.append(t);
  if (R.hamilton) host.append(H.el('div', { class: 'mono' }, 'Reference, non-RL Hamiltonian cycle: ' + R.hamilton.map(h => `${h.size}×${h.size} ${pct5(h.win_rate, 1)}`).join(' · ')));
};

WIDGETS.win_curves = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Greedy test win rate (%) during training, per stage'));
  if (need5(host, H, R.stages && R.stages.length, 'No runs yet.')) return;
  const chart = H.el('div'); host.append(chart);
  const series = [];
  R.stages.forEach((st, i) => st.runs.forEach((r, j) => series.push({
    name: `${st.size}×${st.size}${r.scratch ? ' scratch' : ''}`, x: r.curve.map(c => c[0] * 0.5), y: r.curve.map(c => +(c[2] / (c[3] || 500) * 100).toFixed(1)), color: C5[(i + j) % C5.length],
  })));
  H.lineChart(chart, { series, xlabel: 'env steps (millions, of that stage)', ylabel: 'win rate %', ymin: 0 });
  const c2 = H.el('div'); host.append(H.el('h4', {}, 'Mean fill (% of the cells that can be eaten)'), c2);
  H.lineChart(c2, { series: R.stages.flatMap((st, i) => st.runs.map((r, j) => ({
    name: `${st.size}×${st.size}${r.scratch ? ' scratch' : ''}`, x: r.curve.map(c => c[0] * 0.5), y: r.curve.map(c => +(100 * c[1] / (st.size * st.size - 3)).toFixed(1)), color: C5[(i + j) % C5.length] }))),
    xlabel: 'env steps (millions, of that stage)', ylabel: 'fill %', ymin: 0, h: 240 });
};

WIDGETS.win_replay = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Watch a recorded win (exact re-simulation from seed + actions)'));
  if (need5(host, H, R.replays && R.replays.length, 'No recorded wins yet. The first win of each run is saved automatically.')) return;
  let k = 0, g, t = 0, timer = null, speed = 40;
  const cv = H.el('canvas'), info = H.el('div', { class: 'mono', style: 'white-space:pre' });
  const sel = H.el('select'); R.replays.forEach((r, i) => sel.append(H.el('option', { value: i }, `${r.label}: ${r.width}×${r.width}, seed ${r.seed}, score ${r.score}`)));
  const draw = () => { const r = R.replays[k]; H.board(cv, g, { cell: Math.max(10, Math.floor(300 / g.w)) }); info.textContent = `${r.label}\nstep ${t}/${r.actions.length}  score ${g.score}  length ${g.body.length}${g.done ? '  → ' + g.cause.toUpperCase() : ''}`; };
  const load = () => { stop(); k = +sel.value; t = 0; g = new S.SnakeGame({ width: R.replays[k].width, height: R.replays[k].height, seed: R.replays[k].seed }); draw(); };
  const tick = () => { const r = R.replays[k]; if (g.done || t >= r.actions.length) return stop(); g.step(+r.actions[t++]); draw(); };
  const stop = () => { clearInterval(timer); timer = null; };
  sel.onchange = load;
  host.append(H.el('div', { class: 'row' }, sel, H.el('button', { class: 'btn', onclick: () => { stop(); timer = setInterval(tick, 1000 / speed); } }, '▶ play'),
    H.el('button', { class: 'btn', onclick: stop }, '❚❚'), H.slider('speed ', 5, 200, 5, speed, v => { speed = v; if (timer) { stop(); timer = setInterval(tick, 1000 / speed); } }, v => v + '/s')),
    H.el('div', { class: 'row', style: 'align-items:flex-start' }, cv, info));
  load();
};
