// Node entry point for the engine. The engine itself lives inside index.html (between the
// SNAKE CORE BEGIN/END markers) so the game stays a single self-contained file with no drift.
// Usage: const S = require('./game/snake_core.js');
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/\/\/ === SNAKE CORE BEGIN ===([\s\S]*?)\/\/ === SNAKE CORE END ===/);
if (!m) throw new Error('engine markers not found in index.html');

const mod = { exports: {} };
new Function('module', 'self', m[1])(mod, undefined);
module.exports = mod.exports;
