# Futures board, 2026 (archived)

The Futures hub was the first tab on the site. It had two pages: the **Futures board**
and a link to the playoff projection.

The Futures board compared the model with season-long sportsbook prices from July and
August 2026: national title, make the CFP, conference title, win totals and the Heisman.
Both sides of every comparison were frozen at the preseason. The market side came from
`viz/data/odds.json` (`markets`, `sources`). The model side came from
`viz/data/playoff_preseason.json`, `dynamic.preseason_ratings` in
`viz/data/model_v4.json`, and `viz/data/futures_preseason.json` (the Heisman
candidates' preseason WAR). Below the market panel it showed floor-and-ceiling win
ranges and a sortable season forecast table (Power, projected record, conference
title, CFP, bye, semis, final, title).

It was removed from the site on 23 September 2026. The playoff projection did not go
with it. It is still on the site at **Simulation → Projected bracket**, with its
Current / Preseason toggle.

The last commit with the board live is `f4c16e6` ("v5.1"). The local tag
`futures-2026-archive` points at it.

## What is in this folder

| File | What it holds |
|---|---|
| `futures-board.html` | The hub tab, the section nav and the `#view-dash` section from `viz/index.html`, and the old social-card description |
| `futures-board.js` | Every block removed from `viz/app.js`: the data fetch, the season forecast table, the futures and win-total rows of `BET_RULES`, the board's model side and renderers, the event wiring and the render calls |
| `futures-board.css` | The rules removed from `viz/style.css`, including the media-query lines |

Each block starts with a comment that says where it was.

## What was kept

These files and fields stay in place, so a restore only has to put the code back:

- `viz/data/futures_preseason.json`. The board was its only reader. It is not rebuilt
  by any script, so do not delete it if you may want the Heisman panel back.
- `viz/data/odds.json` `markets` and `sources`. `scripts/export_site_data.py` still
  writes them.
- `viz/data/playoff_preseason.json`. The Projected bracket's "Preseason · locked"
  toggle still reads it, and the Deploy site workflow still caches it.
- `dynamic.preseason_ratings` in `viz/data/model_v4.json`. Week 1 grading and the
  market capture still use it.
- `teamMini()` in `viz/app.js` and `.mini-team` in `viz/style.css`. The Market board
  and Tracking use them.

## How to restore it

The quick way, if nothing else in the three files has changed since the removal
commit: revert that commit.

```
git revert <sha of "Archive the Futures tab ...">
```

The manual way, which works after later edits:

1. In `viz/index.html`, paste the three blocks from `futures-board.html` back where
   their comments say: the hub tab first in `<nav class="hub-nav">`, the section nav
   first in `<div class="section-bar">`, and `<section id="view-dash">` first in
   `<main>`. Optionally put back the old `og:description`.
2. In `viz/app.js`, paste each block from `futures-board.js` back where its comment
   says:
   - add `futuresPreseason` to the end of the destructured list and the
     `futures_preseason.json` fetch to the end of the `Promise.all` array;
   - put the season forecast table block (`RATINGS DASHBOARD`) back just before the
     `PLAYOFF` banner;
   - put the `futures` and `win_total` rows back at the top of `BET_RULES` (leave the
     `spread` and `moneyline` rows as they are, because
     `scripts/export_bet_tracking.py` reads them);
   - put `quantileFromCounts` through `renderFutures()` back right after `BET_RULES`,
     and `renderOutcomeBands()` right after `teamMini()`;
   - put the `#future-market` / `#future-book` listeners back before the
     `#leader-kind` listeners;
   - add `else if (view === "dash") renderDash();` to `render()`, and
     `fillConfSelect()`, `renderDash()`, `renderFutures()` and `renderOutcomeBands()`
     back to `renderAll()`.
3. In `viz/style.css`, paste the blocks from `futures-board.css` back where their
   comments say.
4. Bump the `?v=` numbers on `style.css` and `app.js` in `viz/index.html`.
5. Serve `viz/` (`python -m http.server 8642 -d viz`) and check that the Futures tab
   opens, all five markets render, and the browser console has no errors.

To look at the board exactly as it was, without changing the branch:

```
git worktree add ../futures-2026 futures-2026-archive
python -m http.server 8642 -d ../futures-2026/viz
```
