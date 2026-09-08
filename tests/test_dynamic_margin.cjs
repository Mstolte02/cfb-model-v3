/* The browser's inverse normal must be the same function scipy's ndtri is.

   viz/app.js now builds the projected margin as sigma * normInv(p), so a bad port here
   is a wrong spread on every game rather than a rounding difference somewhere. app.js
   is an IIFE with nothing exported, so the function is lifted out of the source and
   evaluated on its own; the reference values below came from scipy.special.ndtri. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const src = fs.readFileSync(path.join(__dirname, '..', 'viz', 'app.js'), 'utf8');
const found = src.match(/function normInv\(p\) \{[\s\S]*?\n  \}/);
assert(found, 'normInv not found in viz/app.js');
const normInv = eval(`(${found[0]})`);

// [p, scipy.special.ndtri(p)]
const REFERENCE = [
  [0.001, -3.090232306167813], [0.01, -2.326347874040841],
  [0.05, -1.644853626951473], [0.1, -1.281551565544600],
  [0.25, -0.674489750196082], [0.4, -0.253347103135800],
  [0.5, 0.000000000000000], [0.6, 0.253347103135800],
  [0.75, 0.674489750196082], [0.9, 1.281551565544600],
  [0.95, 1.644853626951472], [0.99, 2.326347874040841],
  [0.999, 3.090232306167813],
  // Oklahoma at Michigan on the 2026 board, the game that exposed the split.
  [0.5828145510418772, 0.209099073554620],
];
for (const [p, expected] of REFERENCE) {
  const got = normInv(p);
  assert(Math.abs(got - expected) < 1e-8,
    `normInv(${p}) = ${got}, scipy says ${expected}`);
  // A margin is read to one decimal, so the port must be far inside that at sigma 17.
  assert(Math.abs((got - expected) * 17) < 1e-6, `port drifts a visible point at p=${p}`);
}

// Antisymmetry is what makes a swapped matchup negate the spread exactly rather than
// nearly, so the board cannot favour a team by a different number in each direction.
for (const p of [0.02, 0.17, 0.3, 0.49, 0.5, 0.5828, 0.86, 0.981]) {
  assert(Math.abs(normInv(p) + normInv(1 - p)) < 1e-9, `not antisymmetric at ${p}`);
}
assert.equal(normInv(0.5), 0);

// The clip has to match src/dynamic.py, or Python and the page cap a blowout at
// different spreads.
const clip = src.match(/const MARGIN_P_CLIP = (\.\d+);/);
assert(clip, 'MARGIN_P_CLIP not found in viz/app.js');
assert.equal(Number(clip[1]), 0.001, 'MARGIN_P_CLIP drifted from src/dynamic.py');

/* Weeks already played must keep the margin they were graded on. Recomputing them on
   the current model reads their own results back out of the in-season ratings: applied
   to week 1 it turned a settled 10-4-2 spread record into 14-0-1. These are checks on
   the source rather than on behaviour, because the guarantee is a historical boundary
   that is easy to delete by accident and impossible to notice once it is gone. */
assert(/const MARGIN_BASIS_FROM_WEEK = 2;/.test(src),
  'MARGIN_BASIS_FROM_WEEK is gone or moved: completed weeks would be regraded');
assert(/function predict\(a, b, venue, week\)/.test(src),
  'predict lost its week argument, so nothing can ask for the graded margin');
assert(/week != null && week < MARGIN_BASIS_FROM_WEEK/.test(src),
  'the frozen-margin branch is gone from predict');
assert(/predict\(g\.home, g\.away, "A", g\.week\)/.test(src),
  'marketRows stopped passing the week, so the bet board would regrade played weeks');
assert(/predict\(t, opp, venue, g\.w\)/.test(src),
  'the team schedule stopped passing the week, so it would disagree with the board');

console.log('Dynamic margin: normInv matches scipy ndtri, stays antisymmetric, shares '
  + 'its probability clip with src/dynamic.py, and played weeks stay on the margin they '
  + 'were graded on');
