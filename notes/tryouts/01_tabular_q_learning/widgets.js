// Interactive figures for tryout 01. Each WIDGETS[name](host, R, S, H) gets the results JSON (R), the game
// engine (S, the same code as game/index.html) and the helpers (H) from notes/template.html.

const COLORS = ['#306230', '#c27c0e', '#3a6ea5', '#9b3d8f'];
const ACT = ['straight', 'right', 'left'];
const qRow = (R, idx) => (R.model && R.model.q[idx]) || [0, 0, 0];

// Features as if the head stood at (x, y) with heading d: used for the value heatmap.
function indexAt(S, g, x, y, d) {
  let s = 0;
  [d, (d + 1) % 4, (d + 3) % 4].forEach((dd, i) => { if (g.blocked(x + S.DIRS[dd][0], y + S.DIRS[dd][1])) s |= 1 << i; });
  s |= 1 << (3 + d);
  if (g.food) {
    if (g.food[0] < x) s |= 1 << 7; if (g.food[0] > x) s |= 1 << 8;
    if (g.food[1] < y) s |= 1 << 9; if (g.food[1] > y) s |= 1 << 10;
  }
  return s;
}

function snapshot(g) { return { w: g.w, h: g.h, body: g.body.map(p => [...p]), dir: g.dir, food: g.food && [...g.food] }; }
function restore(S, snap) {
  const g = new S.SnakeGame({ width: snap.w, height: snap.h, seed: 0 });
  g.body = snap.body.map(p => [...p]); g.occ.fill(0);
  for (const [x, y] of g.body) g.occ[y * g.w + x] = 1;
  g.dir = snap.dir; g.food = snap.food && [...snap.food];
  return g;
}

WIDGETS.curves = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Learning curves: greedy evaluation on 500 held-out seeds'));
  const ctl = H.el('div', { class: 'row' }); host.append(ctl);
  const chart = H.el('div'); host.append(chart);
  const note = H.el('div', { class: 'mono' }); host.append(note);
  let board = '12x12', metric = 'eval';
  const draw = () => {
    const sc = R.seedCurves, ep = sc.epochs, run = R.runs[board + '_seed1'].checkpoints;
    const series = metric === 'eval'
      ? Object.entries(sc[board]).map(([k, ys], i) => ({ name: 'eval ' + k, x: ep, y: ys, color: COLORS[i],
          extra: k === 'seed1' ? j => `median ${run[j].median}, p90 ${run[j].p90}, max ${run[j].max}, wins ${run[j].wins}\ndeaths wall/self/timeout ${run[j].wall}/${run[j].self}/${run[j].timeout}` : null }))
      : [{ name: 'eval (greedy)', x: ep, y: run.map(c => c.evalMean), color: COLORS[0] },
         { name: 'train (ε-greedy)', x: ep, y: run.map(c => c.trainMean), color: COLORS[1], dash: '5 4' },
         { name: 'best eval game', x: ep, y: run.map(c => c.max), color: COLORS[2], dash: '2 3' }];
    const hl = board === '12x12' ? [{ y: R.baselines12.greedy, name: 'heuristic 20.4', color: '#b3261e' }] : [];
    H.lineChart(chart, { series, xlabel: 'epoch (×100 episodes)', ylabel: 'score (food eaten)', hlines: hl });
    note.textContent = `win = score ${R.runs[board + '_seed1'].winScore}. Hover points for details.`;
  };
  const sel = (opts, cb) => { const s = H.el('select'); opts.forEach(o => s.append(H.el('option', { value: o[0] }, o[1]))); s.onchange = () => { cb(s.value); draw(); }; return s; };
  ctl.append(H.el('label', {}, 'board ', sel([['12x12', '12×12'], ['6x6', '6×6']], v => board = v)),
             H.el('label', {}, 'view ', sel([['eval', '3 seeds, eval mean'], ['detail', 'seed 1: eval vs train vs best']], v => metric = v)));
  draw();
};

WIDGETS.gamma_horizon = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Discounting: how far ahead does γ look?'));
  const chart = H.el('div'), info = H.el('div', { class: 'mono' });
  let g = 0.95;
  const draw = () => {
    const k = Array.from({ length: 101 }, (_, i) => i);
    H.lineChart(chart, { series: [
      { name: 'weight γ^k', x: k, y: k.map(i => +Math.pow(g, i).toFixed(4)), color: COLORS[0] },
      { name: 'γ = 0.99 ref', x: k, y: k.map(i => +Math.pow(0.99, i).toFixed(4)), color: '#aaa', dash: '4 4' }],
      xlabel: 'steps into the future k', ylabel: 'weight', ymin: 0, ymax: 1, h: 240 });
    const hzn = 1 / (1 - g);
    info.innerHTML = `effective horizon 1/(1−γ) = <b>${hzn.toFixed(1)}</b> steps · food 10 steps away is worth γ¹⁰ = ${Math.pow(g, 10).toFixed(3)} · ` +
      `a death 40 steps away weighs γ⁴⁰ = ${Math.pow(g, 40).toFixed(3)}<br>Snake on 12×12: food ≈ 5–15 steps away; escaping a coil can take 50+ steps.`;
  };
  host.append(H.el('div', { class: 'row' }, H.slider('γ ', 0.5, 0.999, 0.001, g, v => { g = v; draw(); }, v => v.toFixed(3))), chart, info);
  draw();
};

WIDGETS.q_update = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'One Q-learning update, on a real Snake transition'));
  const q = {}; // local editable copy of the learned table
  const reset = () => { for (const k in q) delete q[k]; Object.entries(R.model.q).forEach(([k, v]) => q[k] = [...v]); };
  reset();
  let alpha = 0.1, gamma = 0.95, seed = 100003, g, eps = 0.0, rng = new S.Rng(1);
  const cv = H.el('canvas'), out = H.el('div', { class: 'mono', style: 'white-space:pre;line-height:1.5' });
  const row = i => q[i] || (q[i] = [0, 0, 0]);
  const newGame = () => { g = new S.SnakeGame({ seed }); render(null); };
  const render = (t) => {
    H.board(cv, g, { cell: 18 });
    const s = S.basicStateIndex(g), f = S.basicFeatures(g);
    let txt = `state s = row ${s}   features ${f.join('')}  (danger S/R/L | heading R D L U | food L R U D)\n` +
              `Q(s,·) = [${row(s).map(v => v.toFixed(3)).join(', ')}]   greedy → ${ACT[S.argmax(row(s))]}\n`;
    if (t) {
      txt += `\nlast transition: a = ${ACT[t.a]}${t.explore ? ' (random, ε)' : ''},  r = ${t.r},  ${t.term ? 'TERMINAL → no bootstrap' : `max_a' Q(s',a') = ${t.next.toFixed(3)}`}\n` +
             `target  = r + γ·max Q(s',·) = ${t.r} + ${gamma.toFixed(2)}·${t.term ? 0 : t.next.toFixed(3)} = ${t.target.toFixed(3)}\n` +
             `δ (TD error) = target − Q(s,a) = ${t.target.toFixed(3)} − ${t.old.toFixed(3)} = ${(t.target - t.old).toFixed(3)}\n` +
             `Q(s,a) ← ${t.old.toFixed(3)} + α·δ = ${t.old.toFixed(3)} + ${alpha.toFixed(2)}·${(t.target - t.old).toFixed(3)} = ${t.new.toFixed(3)}`;
    }
    out.textContent = txt;
  };
  const step = () => {
    if (g.done) { g.reset(); render(null); return; }
    const s = S.basicStateIndex(g), explore = rng.nextU32() / 4294967296 < eps;
    const a = explore ? rng.int(3) : S.argmax(row(s));
    const r = g.step(a);
    const next = r.terminated ? 0 : Math.max(...row(S.basicStateIndex(g)));
    const old = row(s)[a], target = r.reward + (r.terminated ? 0 : gamma * next);
    row(s)[a] = old + alpha * (target - old);
    render({ a, r: r.reward, term: r.terminated, next, old, target, new: row(s)[a], explore });
  };
  const btn = (t, f) => H.el('button', { class: 'btn', onclick: f }, t);
  host.append(
    H.el('div', { class: 'row' }, H.slider('α ', 0.01, 1, 0.01, alpha, v => alpha = v, v => v.toFixed(2)),
      H.slider('γ ', 0, 0.999, 0.001, gamma, v => gamma = v, v => v.toFixed(3)),
      H.slider('ε ', 0, 1, 0.05, eps, v => eps = v, v => v.toFixed(2))),
    H.el('div', { class: 'row' }, btn('step + update', step), btn('10 steps', () => { for (let i = 0; i < 10; i++) step(); }),
      btn('new episode', () => { seed++; newGame(); }), btn('restore learned table', () => { reset(); render(null); })),
    H.el('div', { class: 'row', style: 'align-items:flex-start' }, cv, out));
  newGame();
};

WIDGETS.epsilon_schedule = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'ε-greedy schedule, and what exploration costs'));
  const c = R.config, run = R.runs['12x12_seed1'].checkpoints;
  let e0 = c.epsStart, e1 = c.epsEnd, frac = c.epsDecayEpochs / c.epochs;
  const chart = H.el('div'), info = H.el('div', { class: 'mono' });
  const draw = () => {
    const ep = Array.from({ length: 41 }, (_, i) => i * 5);
    const eps = ep.map(x => +Math.max(e1, e0 - (e0 - e1) * x / (c.epochs * frac)).toFixed(3));
    H.lineChart(chart, { series: [{ name: 'ε (your schedule)', x: ep, y: eps, color: COLORS[2] }], xlabel: 'epoch', ylabel: 'ε', ymin: 0, ymax: 1, h: 200 });
    const p = n => (100 * Math.pow(1 - e1, n)).toFixed(1);
    info.innerHTML = `At the floor ε = ${e1}: chance an episode of 200 / 500 moves has <i>no</i> random action = ${p(200)}% / ${p(500)}%.<br>` +
      `Measured (run used ε 1→0.01 over 100 epochs): epoch 200 train mean ${run[8].trainMean} vs greedy eval ${run[8].evalMean}.`;
  };
  host.append(H.el('div', { class: 'row' },
    H.slider('start ', 0, 1, 0.05, e0, v => { e0 = v; draw(); }, v => v.toFixed(2)),
    H.slider('floor ', 0, 0.3, 0.005, e1, v => { e1 = v; draw(); }, v => v.toFixed(3)),
    H.slider('decay over ', 0.05, 1, 0.05, frac, v => { frac = v; draw(); }, v => Math.round(v * 100) + '% of run')), chart, info);
  draw();
};

WIDGETS.aliasing = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'State aliasing: one table row, two very different situations'));
  // Search real greedy episodes for two states with the same row where the greedy move is safe in one
  // and walks into a pocket smaller than the snake in the other.
  const pol = g => S.argmax(qRow(R, S.basicStateIndex(g)));
  const seen = {};
  let pair = null;
  for (let seed = 100000; seed < 100200 && !pair; seed++) {
    const g = new S.SnakeGame({ seed });
    while (!g.done && !pair) {
      if (g.body.length > 12) {
        const idx = S.basicStateIndex(g), a = pol(g);
        const d = a === 0 ? g.dir : a === 1 ? (g.dir + 1) % 4 : (g.dir + 3) % 4;
        const nx = g.body[0][0] + S.DIRS[d][0], ny = g.body[0][1] + S.DIRS[d][1];
        const area = S.reachable(g, nx, ny), trap = area < g.body.length;
        const snap = { ...snapshot(g), a, area, nx, ny, idx };
        const other = seen[idx] && seen[idx][trap ? 'safe' : 'trap'];
        if (other) pair = trap ? [other, snap] : [snap, other];
        (seen[idx] = seen[idx] || {})[trap ? 'trap' : 'safe'] = seen[idx][trap ? 'trap' : 'safe'] || snap;
      }
      g.step(pol(g));
    }
  }
  if (!pair) { host.append('No aliased pair found in 200 episodes.'); return; }
  const row = H.el('div', { class: 'row', style: 'align-items:flex-start' });
  pair.forEach((snap, k) => {
    const g = restore(S, snap);
    const region = new Set();
    // flood fill from the chosen cell to colour the space the greedy move leads into
    const stack = [[snap.nx, snap.ny]];
    while (stack.length) {
      const [x, y] = stack.pop(), key = y * g.w + x;
      if (region.has(key) || g.blocked(x, y)) continue;
      region.add(key);
      S.DIRS.forEach(([dx, dy]) => stack.push([x + dx, y + dy]));
    }
    const cv = H.el('canvas');
    H.board(cv, g, { cell: 16, overlay: (x, y) => region.has(y * g.w + x) ? (k ? 'rgba(179,38,30,.35)' : 'rgba(30,122,52,.3)') : null });
    row.append(H.el('div', {}, cv, H.el('div', { class: 'mono' },
      `${k ? 'TRAP' : 'SAFE'}: row ${snap.idx}, greedy → ${ACT[snap.a]}`), H.el('div', { class: 'mono' },
      `room after the move: ${snap.area} cells, snake length ${snap.body.length}`)));
  });
  host.append(row, H.el('div', { class: 'mono', style: 'margin-top:6px' },
    `Both boards produce features ${pair[0].idx.toString(2).padStart(11, '0').split('').reverse().join('')} → identical Q(s,·) = [${qRow(R, pair[0].idx).map(v => v.toFixed(2)).join(', ')}]. ` +
    'No table entry can be right for both; the coloured region is what the agent cannot see.'));
};

WIDGETS.checkpoint_table = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Checkpoint log: every 25 epochs (2,500 episodes)'));
  const sel = H.el('select'); ['12x12', '6x6'].forEach(b => sel.append(H.el('option', { value: b }, b.replace('x', '×'))));
  const box = H.el('div');
  const draw = () => {
    const run = R.runs[sel.value + '_seed1'], pct = (c, k) => Math.round(100 * c[k] / 500) + '%';
    let prev = null;
    box.innerHTML = '<table><tr><th>epoch</th><th>ε</th><th>train mean (max, wins)</th><th>eval mean</th><th>median</th><th>p90</th><th>max</th><th>wins/500</th><th>Δ eval</th><th>wall / self / timeout</th></tr>' +
      run.checkpoints.map(c => {
        const d = prev == null ? '–' : (c.evalMean - prev >= 0 ? '+' : '') + (c.evalMean - prev).toFixed(2); prev = c.evalMean;
        return `<tr><td>${c.epoch}</td><td>${c.eps}</td><td>${c.trainMean == null ? '–' : `${c.trainMean} (${c.trainMax}, ${c.trainWins})`}</td><td><b>${c.evalMean}</b></td><td>${c.median}</td><td>${c.p90}</td><td>${c.max}</td><td>${c.wins}</td><td>${d}</td><td>${pct(c, 'wall')} / ${pct(c, 'self')} / ${pct(c, 'timeout')}</td></tr>`;
      }).join('') + `</table><div class="mono">win = score ${run.winScore}</div>`;
  };
  sel.onchange = draw;
  host.append(H.el('div', { class: 'row' }, H.el('label', {}, 'board ', sel)), box);
  draw();
};

WIDGETS.sweep_heatmap = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'γ × α sweep: mean greedy score (3 seeds, hover for each)'));
  const sw = R.sweep, cw = 120, ch = 36, m = 60;
  const svg = H.s('svg', { viewBox: `0 0 ${m + cw * sw.alphas.length + 20} ${m + ch * sw.gammas.length}`, width: '480' });
  sw.gammas.forEach((gm, i) => {
    const t = H.s('text', { x: m - 8, y: m + i * ch + ch / 2 + 4, 'text-anchor': 'end' }); t.textContent = 'γ ' + gm; svg.append(t);
    sw.alphas.forEach((al, j) => {
      if (i === 0) { const t2 = H.s('text', { x: m + j * cw + cw / 2, y: m - 10, 'text-anchor': 'middle' }); t2.textContent = 'α ' + al; svg.append(t2); }
      const v = sw.means[i][j], mean = v.reduce((a, b) => a + b, 0) / v.length;
      const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1));
      const r = H.s('rect', { x: m + j * cw, y: m + i * ch, width: cw - 2, height: ch - 2, fill: H.heat(mean, 10, 23), rx: 3 });
      r.onmousemove = ev => H.showTip(`γ=${gm}, α=${al}\nseeds: ${v.join(', ')}\nmean ${mean.toFixed(2)} ± ${sd.toFixed(2)} (sd)`, ev);
      r.onmouseleave = H.hideTip;
      const t3 = H.s('text', { x: m + j * cw + cw / 2, y: m + i * ch + ch / 2 + 4, 'text-anchor': 'middle' });
      t3.textContent = mean.toFixed(1) + ' ± ' + sd.toFixed(1); t3.setAttribute('style', 'fill:#0f380f;font-weight:600');
      svg.append(r, t3);
    });
  });
  host.append(svg, H.el('div', { class: 'mono' }, sw.note + '. Red = worse, green = better.'));
};

WIDGETS.replay = (host, R, S, H) => {
  host.append(H.el('h4', {}, 'Replay the trained table on held-out seeds, with a live value heatmap'));
  let seed = 100000, g, timer = null, speed = 12, showHeat = true;
  const cv = H.el('canvas'), info = H.el('div', { class: 'mono', style: 'white-space:pre' });
  const pol = gm => S.argmax(qRow(R, S.basicStateIndex(gm)));
  const draw = () => {
    let lo = Infinity, hi = -Infinity; const vals = {};
    if (showHeat) for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++) {
      if (g.occ[y * g.w + x]) continue;
      const v = Math.max(...qRow(R, indexAt(S, g, x, y, g.dir))); vals[y * g.w + x] = v; lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    H.board(cv, g, { cell: 24, overlay: showHeat ? (x, y) => (y * g.w + x) in vals ? H.heat(vals[y * g.w + x], lo, hi) : null : null });
    const q = qRow(R, S.basicStateIndex(g));
    info.textContent = `seed ${seed}   score ${g.score}   length ${g.body.length}   steps ${g.steps}${g.done ? '   → ' + g.cause.toUpperCase() : ''}\n` +
      `Q(s,·): straight ${q[0].toFixed(3)}  right ${q[1].toFixed(3)}  left ${q[2].toFixed(3)}   → ${ACT[S.argmax(q)]}` +
      (showHeat ? `\nheatmap: max_a Q with the head moved to each cell (range ${lo.toFixed(2)} … ${hi.toFixed(2)})` : '');
  };
  const load = () => { stop(); g = new S.SnakeGame({ seed }); draw(); };
  const tick = () => { if (g.done) return stop(); g.step(pol(g)); draw(); };
  const stop = () => { clearInterval(timer); timer = null; };
  const play = () => { stop(); timer = setInterval(tick, 1000 / speed); };
  const seedIn = H.el('input', { type: 'number', value: seed, style: 'width:90px' });
  seedIn.onchange = () => { seed = +seedIn.value; load(); };
  const chk = H.el('input', { type: 'checkbox', checked: '' }); chk.onchange = () => { showHeat = chk.checked; draw(); };
  host.append(H.el('div', { class: 'row' }, H.el('label', {}, 'eval seed ', seedIn),
    H.el('button', { class: 'btn', onclick: play }, '▶ play'), H.el('button', { class: 'btn', onclick: stop }, '❚❚'),
    H.el('button', { class: 'btn', onclick: () => { stop(); tick(); } }, 'step'),
    H.el('button', { class: 'btn', onclick: () => { seedIn.value = ++seed; load(); } }, 'next seed'),
    H.slider('speed ', 2, 60, 1, speed, v => { speed = v; if (timer) play(); }, v => v + '/s'),
    H.el('label', {}, chk, 'value heatmap')), H.el('div', { class: 'row', style: 'align-items:flex-start' }, cv, info));
  load();
};
