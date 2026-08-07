/* CFB Model v3 — 2026 visuals.
   Tabs: Season Odds · Playoff Projection · Scenario Builder · Matchup Simulator ·
   Team Breakdown · Team Ratings · Players.

   The trained model math is frozen (coefficients in model*.json) and is a verified
   port of the Python model — win probability, margin and points agree with the
   backend to 4 decimals. Every number on screen derives from those coefficients
   plus the exported per-team feature vectors, so the bracket, the schedule rows and
   the matchup simulator cannot disagree with each other. */
(async function () {
  // no-cache, not no-store: still uses the HTTP cache but revalidates first. The
  // data files are rewritten every time the pipeline runs, and a stale copy shows
  // wrong numbers with no visible error.
  const fetchJSON = u => fetch(u, { cache: "no-cache" }).then(r => r.json());

  /* ---------- the data ----------
     ONE BUILD. The site used to ship two side by side with a header toggle - the
     default and a PFF-only one that took no outside opinion anywhere - and the
     reader had to pick. There is one now: our WAR, with EA's ORDERING substituted
     only for players under 300 prior snaps, and the starting quarterbacks ordered by
     the five-source composite in qbs_2026.xlsx. Proven players keep our own number.

     diagnostics.json is not fetched: the Method page was its only reader, and pulling
     25KB on every load to render nothing is a cost with no page behind it.
     scripts/export_diagnostics.py still writes the file. */
  const [teams, schedule, players, ratings, playoff, model] = await Promise.all([
    fetchJSON("data/teams.json"),
    fetchJSON("data/schedule.json"),
    fetchJSON("data/players.json").catch(() => ({})),
    fetchJSON("data/ratings.json"),
    fetchJSON("data/playoff.json"),
    fetchJSON("data/model.json"),
  ]);
  // An older lens toggle offered a roster-weighted variant that leaned harder on the
  // two-deep; it was a knowingly worse backtest kept as an alternative view, and it is
  // gone too. Talent is a PFF / recruiting / WAR blend whose weights are swept jointly
  // under leave-one-season-out and exported, not written in here.
  const DATA = { ratings, playoff, model };
  const cur = () => DATA;

  // The header used to carry a "TALENT 38/38/25" chip. It is gone: the blend is an
  // internal weighting that means nothing to a reader who has not been told what the
  // three sources are, and it sat in the corner of every page. The Method page
  // explains it once, in words, where someone has actually asked.

  /* ---------- team metadata ---------- */
  const meta = teams;
  const logoURL = t => (meta[t] && meta[t].logo) || "";
  const conf = t => (meta[t] && meta[t].conference) || "—";
  const abbr = t => (meta[t] && meta[t].abbreviation) || t;
  const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/"/g, "&quot;");

  /* Brand colour, lightened until it reads as text on the dark panel. rgba() keeps
     the true brand hue for washes and fills, where contrast isn't the constraint. */
  const rawColor = t => (meta[t] && meta[t].color) || "#58a6ff";
  function color(t) {
    const c = rawColor(t);
    let [r, g, b] = [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
    if (isNaN(r)) return "#58a6ff";
    while (0.299 * r + 0.587 * g + 0.114 * b < 100) {
      r = Math.min(255, r + 32); g = Math.min(255, g + 32); b = Math.min(255, b + 32);
    }
    return `rgb(${r},${g},${b})`;
  }
  function rgba(t, a) {
    const c = rawColor(t);
    const [r, g, b] = [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
    return isNaN(r) ? `rgba(88,166,255,${a})` : `rgba(${r},${g},${b},${a})`;
  }

  const P4 = new Set(["ACC", "Big 12", "Big Ten", "SEC"]);
  const G6 = new Set(["American Athletic", "Conference USA", "Mid-American",
    "Mountain West", "Pac-12", "Sun Belt"]);

  const baseVec = t => cur().model.teams[t];
  const rankNames = () => liveRatings().map(t => t.team);
  const ratingRow = () => Object.fromEntries(liveRatings().map(t => [t.team, t]));
  const simRow = () => Object.fromEntries((cur().playoff.teams || []).map(t => [t.team, t]));

  /* ---------- what-if: edit a player's WAR, re-derive the ratings ----------
     Talent is a retired feature with a coefficient of exactly zero, so an edit does
     NOT reach a prediction through the talent slot. It reaches one through the shrink
     in matchup.team_frame, which pulls a team toward its talent-implied rating:

         O_adj = (1 - lam*u)*O + lam*u*(b_o*talent)

     That is linear in talent, so a talent delta moves the exported vector by exactly
     lam*u*b_o*dt — no need to reconstruct the pre-shrink O. WAR is one axis of talent
     at config.WAR_BLEND, and it enters as a z-score taken across the teams the WAR
     build covers. Editing one player therefore shifts the league mean and sd and moves
     every other team a little too, which is not a rounding artefact: it is what the
     model would do if that player really were that good.

     What this CANNOT update is the playoff odds. Those come from a 20,000-season Monte
     Carlo over the real schedule, run in Python. Re-running it here would be a
     different simulation pretending to be the same one, so the odds are left showing
     their unedited values and the UI says so rather than quietly implying otherwise. */
  const WI = (function () {
    const cfg = (model && model.whatif) || null;
    const edits = new Map();                       // key: team + NUL + player -> new WAR
    const KEY = "cfb-whatif-v1";
    // NUL joins the key, not a space: "Ohio State" and "Julian Sayin" both contain
    // spaces, so a space-joined key split into four pieces and named the wrong team.
    const k = (t, p) => t + "\u0000" + p;

    const rosterOf = t => (players[t] && players[t].players) || null;
    // Base summed WAR per team. players.json carries `total` already, but it is
    // rounded to 3dp for transport, so it is re-summed from the rows the page will
    // actually edit - otherwise a scenario with no edits would not reproduce zero.
    const baseWar = {};
    for (const [t, r] of Object.entries(players)) {
      const rs = r && r.players;
      if (rs) baseWar[t] = rs.reduce((s, p) => s + (p.raw || 0), 0);
      else if (r && r.total != null) baseWar[t] = r.total;
    }

    const popZ = obj => {                          // ddof=0, matching war.talent_by_year
      const ts = Object.keys(obj), v = ts.map(t => obj[t]);
      if (!v.length) return {};
      const mu = v.reduce((a, b) => a + b, 0) / v.length;
      const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) * (b - mu), 0) / v.length);
      return Object.fromEntries(ts.map((t, i) => [t, sd ? (v[i] - mu) / sd : 0]));
    };
    const zBase = popZ(baseWar);

    let delta = {};                                // team -> talent delta
    let ver = 0;                                   // bumped on every change, for caches
    function recompute() {
      ver++;
      delta = {};
      if (!cfg || !edits.size) return;
      const now = Object.assign({}, baseWar);
      for (const [key, val] of edits) {
        const [t, name] = key.split("\u0000");
        const rs = rosterOf(t);
        if (!rs || now[t] == null) continue;
        const p = rs.find(x => x.n === name);
        if (p) now[t] += val - (p.raw || 0);
      }
      const zNow = popZ(now);
      for (const t of Object.keys(zNow)) delta[t] = cfg.warBlend * (zNow[t] - zBase[t]);
    }

    function vec(t) {
      const v = baseVec(t);
      if (!v || !cfg || !edits.size) return v;
      const dt = delta[t];
      if (!dt) return v;
      const s = cfg.lam * (cfg.u[t] != null ? cfg.u[t] : 1);
      const out = v.slice();
      out[0] += s * cfg.bO * dt;
      out[1] += s * cfg.bD * dt;
      return out;
    }

    function save() {
      try {
        localStorage.setItem(KEY, JSON.stringify([...edits]));
      } catch (e) { /* private mode; the scenario just will not outlive the tab */ }
    }
    function load() {
      try {
        const raw = localStorage.getItem(KEY);
        if (raw) for (const [key, v] of JSON.parse(raw)) edits.set(key, v);
      } catch (e) { edits.clear(); }
      recompute();
    }

    return {
      enabled: () => !!cfg,
      vec, count: () => edits.size, version: () => ver,
      teamWar: t => {
        let w = baseWar[t] || 0;
        const rs = rosterOf(t);
        if (rs) for (const p of rs) {
          const e = edits.get(k(t, p.n));
          if (e != null) w += e - (p.raw || 0);
        }
        return w;
      },
      baseTeamWar: t => baseWar[t] || 0,
      talentDelta: t => delta[t] || 0,
      get: (t, p) => edits.get(k(t, p)),
      set(t, p, v) {
        const rs = rosterOf(t), row = rs && rs.find(x => x.n === p);
        if (!row) return;
        if (v == null || !isFinite(v) || Math.abs(v - (row.raw || 0)) < 1e-9) {
          edits.delete(k(t, p));
        } else {
          edits.set(k(t, p), v);
        }
        recompute(); save();
      },
      reset() { edits.clear(); recompute(); save(); },
      editedTeams() {
        const s = new Set();
        for (const key of edits.keys()) s.add(key.split("\u0000")[0]);
        return s;
      },
      init: load,
    };
  })();
  WI.init();

  const vecOf = t => WI.vec(t);

  /* Power = mean neutral-site win probability against every other rated team. It is
     recomputed here rather than read from ratings.json whenever a scenario is active,
     because an edited team has to move in the table it is ranked in.

     verifyPower() below is the guard that makes that safe: with no edits this routine
     has to reproduce the exported column, and if it ever stops doing so the page is
     showing a power rating the backend never computed. It logs loudly instead of
     quietly disagreeing. */
  let powerCache = null, powerVer = -1;
  function livePower() {
    if (powerVer === WI.version() && powerCache) return powerCache;
    const names = cur().ratings.teams.map(r => r.team);
    const V = names.map(vecOf);
    const out = {};
    for (let i = 0; i < names.length; i++) {
      let s = 0;
      for (let j = 0; j < names.length; j++) {
        if (i !== j) s += winpFromDiff(diffVec(V[i], V[j]), 0);
      }
      out[names[i]] = s / (names.length - 1);
    }
    powerCache = out; powerVer = WI.version();
    return out;
  }

  function verifyPower() {
    const P = livePower();
    let worst = 0, who = "";
    for (const r of cur().ratings.teams) {
      const d = Math.abs(P[r.team] - r.power);
      if (d > worst) { worst = d; who = r.team; }
    }
    if (worst > 5e-4) {
      console.warn(`[what-if] client power differs from ratings.json by ${worst.toFixed(5)} ` +
                   `(worst: ${who}). Edits would move teams against a baseline the ` +
                   `backend did not produce.`);
    }
    return worst;
  }

  /* Ratings rows with everything an edit touches re-derived when a scenario is active.
     O, D and talent move as well as power and rank: the ratings table reads all of
     them from here, and a page that showed a team climbing while its offense and talent
     sat unchanged would be contradicting itself on screen. */
  function liveRatings() {
    const rows = cur().ratings.teams;
    if (!WI.enabled() || !WI.count()) return rows;
    const P = livePower(), cfg = cur().model.whatif;
    const out = rows.map(r => {
      const dt = WI.talentDelta(r.team);
      const s = cfg.lam * (cfg.u[r.team] != null ? cfg.u[r.team] : 1);
      return Object.assign({}, r, {
        power: P[r.team],
        O: r.O != null ? r.O + s * cfg.bO * dt : r.O,
        D: r.D != null ? r.D + s * cfg.bD * dt : r.D,
        talent: r.talent != null ? r.talent + dt : r.talent,
      });
    });
    const order = out.slice().sort((a, b) => b.power - a.power);
    order.forEach((r, i) => { r.rank = i + 1; });
    return out;
  }

  /* ---------- frozen win-prob math ---------- */
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
      0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const normCdf = x => 0.5 * (1 + erf(x / Math.SQRT2));
  const dot = (c, v) => c[0]*v[0]+c[1]*v[1]+c[2]*v[2]+c[3]*v[3]+c[4]*v[4]+c[5]*v[5];
  function winpFromDiff(x, homeA) {
    const M = cur().model, L = M.logistic, G = M.margin;
    const z = L.intercept + dot(L.coef, x) + homeA * L.hfa;
    const m = G.intercept + dot(G.coef, x) + homeA * G.hfa;
    return M.ens_w * sigmoid(z) + (1 - M.ens_w) * normCdf(m / G.sigma);
  }
  const diffVec = (a, b) => [a[0]-b[1], a[1]-b[0], a[2]-b[2], a[3]-b[3], a[4]-b[4], a[5]-b[5]];

  /* One prediction routine for every view. venue is "A" | "B" | "N", A's perspective. */
  function predict(a, b, venue) {
    const A = vecOf(a), B = vecOf(b), M = cur().model;
    if (!A || !B) return null;
    const homeA = venue === "A" ? 1 : 0, homeB = venue === "B" ? 1 : 0;
    let pA, marginA;
    if (homeB) {
      const xb = diffVec(B, A);
      pA = 1 - winpFromDiff(xb, 1);
      marginA = -(M.margin.intercept + dot(M.margin.coef, xb) + M.margin.hfa);
    } else {
      const x = diffVec(A, B);
      pA = winpFromDiff(x, homeA);
      marginA = M.margin.intercept + dot(M.margin.coef, x) + homeA * M.margin.hfa;
    }
    const Pt = M.points;
    const ptsA = Pt.intercept + Pt.coef[0]*A[0] + Pt.coef[1]*B[1] + Pt.coef[2]*homeA;
    const ptsB = Pt.intercept + Pt.coef[0]*B[0] + Pt.coef[1]*A[1] + Pt.coef[2]*homeB;
    const total = ptsA + ptsB;
    return { pA, margin: marginA, total,
             scoreA: (total + marginA) / 2, scoreB: (total - marginA) / 2 };
  }

  /* ---------- displayed scoreline ----------
     Each side of the projected score, rounded. This used to snap to the nearest
     REAL scoreline from 2021-25, on the argument that football scores are built out
     of 7s and 3s so 27-24 is likelier than the 28-24 that rounding gives. That was
     true about the scoreline and wrong about everything next to it: snapping moves
     the implied margin off the model's own spread, so the projected score and the
     spread printed beside it disagreed about the same game. The spread is the
     number people read, so the scoreline defers to it.

     The winner still has to agree with the win probability. Those two can diverge on
     a near-coin-flip, because pA blends the logistic with the margin model while the
     score comes from the margin alone: a 51% favourite can carry a margin of -0.2.
     Left alone that put a team in the championship game after losing its semifinal on
     the scoreboard. Football has no ties either, so a level game gives the favourite
     the extra point. */
  const SHAPE = cur().model.shape || null;
  function displayScore(r, winnerIsA) {
    if (winnerIsA === undefined) winnerIsA = r.pA >= 0.5;
    let a = Math.max(0, Math.round(r.scoreA));
    let b = Math.max(0, Math.round(r.scoreB));
    if (winnerIsA && a <= b) a = b + 1;
    else if (!winnerIsA && b <= a) b = a + 1;
    return [a, b];
  }

  /* P(|margin| = k) from the fitted distribution of actual margins given the
     predicted one. The normal the win model uses puts .044 on a 3-point margin when
     the real figure is .106, so this is the only honest source for a key number. */
  function marginPMF(margin) {
    if (!SHAPE) return null;
    const P = SHAPE.margin_pmf;
    const i = Math.max(0, Math.min(P.rows.length - 1,
      Math.round((margin - P.pred_lo) / P.pred_step)));
    return P.rows[i];
  }
  function keyNumberP(margin, k) {
    const row = marginPMF(margin);
    if (!row) return null;
    const P = SHAPE.margin_pmf;
    let s = 0;
    for (let j = 0; j < row.length; j++) {
      if (Math.abs(P.margin_lo + j) === k) s += row[j];
    }
    return s;
  }

  /* A margin under half a point is a coin flip, and "−0.0" reads as a rendering
     bug rather than a pick'em. */
  function spreadLabel(margin) {
    if (Math.abs(margin) < 0.5) return `<span class="pk">PK</span>`;
    // en dash rather than the minus sign, which sets far too wide next to figures
    const sign = margin >= 0 ? "–" : "+";
    return `<span class="spread ${margin >= 0 ? "pos" : "neg"}"><span class="sgn">${sign}</span>${Math.abs(margin).toFixed(1)}</span>`;
  }

  /* ---------- tabs ---------- */
  document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === b));
    document.querySelectorAll(".view").forEach(v =>
      v.classList.toggle("active", v.id === "view-" + b.dataset.view));
    render(b.dataset.view);
  }));

  /* =======================================================================
     RATINGS DASHBOARD
     ======================================================================= */
  /* This page used to show Offense, Defense, Talent, Returning and SOS. All five are
     on the Team Breakdown page in more detail than a table cell can carry, and having
     them here meant the odds - the thing the model exists to produce - were buried on
     a second tab behind the bracket. So the columns are the odds now, and the ratings
     that produce them stay one click away.

     Power survives because it is what the ranking is built on and because it is the
     one column that moves when a Proj WAR is edited on the Players tab; drop it and
     that whole feedback loop becomes invisible. */
  const COLS = [
    { k: "rank",       h: "#",            n: true },
    { k: "team",       h: "Team",         n: false },
    { k: "power",      h: "Power",        n: true },
    { k: "record",     h: "Proj Record",  n: true, sort: r => r.avg_wins ?? -1,
      fmt: r => r.avg_wins != null ? `${r.avg_wins.toFixed(1)}–${r.avg_losses.toFixed(1)}` : "—" },
    { k: "conf_champ", h: "Conf Champ",   n: true, bar: "green",
      fmt: r => pct(r.conf_champ ?? 0, 0) },
    { k: "playoff",    h: "Make CFP",     n: true, bar: "tint",
      fmt: r => pct(r.playoff ?? 0, 0) },
    { k: "bye",        h: "Bye",          n: true, fmt: r => pct(r.bye ?? 0, 0) },
    { k: "sf",         h: "Semis",        n: true, fmt: r => pct(r.sf ?? 0, 0) },
    { k: "final",      h: "Final",        n: true, fmt: r => pct(r.final ?? 0, 0) },
    { k: "champ",      h: "Natty",        n: true, fmt: r => pct(r.champ ?? 0, 1) },
  ];
  let sortKey = "power", sortDesc = true;

  function dashRows() {
    const q = document.getElementById("dash-search").value.trim().toLowerCase();
    const c = document.getElementById("dash-conf").value;
    const tier = document.getElementById("dash-tier").value;
    let rows = liveRatings().slice();
    if (q) rows = rows.filter(r => r.team.toLowerCase().includes(q) ||
      ((meta[r.team] && meta[r.team].mascot) || "").toLowerCase().includes(q));
    if (c) rows = rows.filter(r => r.conference === c);
    if (tier === "p4") rows = rows.filter(r => P4.has(r.conference));
    if (tier === "g6") rows = rows.filter(r => G6.has(r.conference));
    const col = COLS.find(x => x.k === sortKey) || COLS[2];
    const val = col.sort || (r => r[sortKey]);
    rows.sort((a, b) => {
      if (sortKey === "team") {
        return sortDesc ? b.team.localeCompare(a.team) : a.team.localeCompare(b.team);
      }
      return sortDesc ? val(b) - val(a) : val(a) - val(b);
    });
    return rows;
  }

  function renderDash() {
    const rows = dashRows();
    const all = liveRatings();
    const maxPower = Math.max(...all.map(r => r.power));
    // Playoff odds are scaled to the leader rather than to 100%, because nobody is
    // near certain and a raw percentage scale leaves every bar a stub.
    const maxCFP = Math.max(...all.map(r => r.playoff || 0), 0.01);
    document.getElementById("dash-count").textContent =
      `${rows.length} of ${all.length} teams`;

    const head = COLS.map(c => {
      const on = c.k === sortKey;
      return `<th class="${c.n ? "num" : ""} sortable${on ? " sorted" : ""}"
        data-k="${c.k}">${c.h}${on ? (sortDesc ? " ▾" : " ▴") : ""}</th>`;
    }).join("");

    const body = rows.map(r => {
      const t = r.team, tint = color(t);
      const cells = COLS.map(c => {
        if (c.k === "team") return `<td><div class="team-cell">
            <span class="team-stripe" style="background:${tint}"></span>
            <img src="${logoURL(t)}" alt="" loading="lazy">
            <div><button class="team-link" data-team="${esc(t)}">${esc(t)}</button>
              <div class="conf">${esc(r.conference)}</div></div></div></td>`;
        if (c.k === "rank") return `<td class="rank">${r.rank}</td>`;
        // Number first, fill second: .bar-wrap stacks them, so the source order is
        // what puts the mini-bar UNDER the figure rather than over it.
        if (c.k === "power") return `<td class="num"><div class="bar-wrap">
            <span class="pct">${(100 * r.power).toFixed(1)}</span>
            <div class="bar"><i style="width:${100 * r.power / maxPower}%;background:${tint}"></i></div>
          </div></td>`;
        if (c.bar) {
          const v = r[c.k] || 0;
          // Conference titles are a share of one trophy and read naturally against
          // 100%; playoff odds are scaled to the leader instead (see maxCFP above).
          const green = c.bar === "green";
          const w = green ? 100 * v : 100 * v / maxCFP;
          const style = `width:${w}%${green ? "" : `;background:${tint}`}`;
          return `<td class="num"><div class="bar-wrap">
              <span class="pct">${c.fmt(r)}</span>
              <div class="bar"><i class="${green ? "green" : ""}" style="${style}"></i></div>
            </div></td>`;
        }
        return `<td class="num">${c.fmt ? c.fmt(r) : r[c.k]}</td>`;
      }).join("");
      return `<tr class="${r.rank <= 4 ? "top4" : ""}">${cells}</tr>`;
    }).join("");

    document.getElementById("dash-table").innerHTML =
      `<table class="dash"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

    document.querySelectorAll("#dash-table th.sortable").forEach(th =>
      th.addEventListener("click", () => {
        const k = th.dataset.k;
        if (k === sortKey) sortDesc = !sortDesc;
        else { sortKey = k; sortDesc = !(k === "rank" || k === "team"); }
        renderDash();
      }));
    wireTeamLinks();
  }
  ["dash-search", "dash-conf", "dash-tier"].forEach(id =>
    document.getElementById(id).addEventListener("input", renderDash));

  function fillConfSelect() {
    const keep = document.getElementById("dash-conf").value;
    const cs = [...new Set(cur().ratings.teams.map(t => t.conference))].sort();
    const sel = document.getElementById("dash-conf");
    sel.innerHTML = `<option value="">All conferences</option>` +
      cs.map(c => `<option value="${esc(c)}"${c === keep ? " selected" : ""}>${esc(c)}</option>`).join("");
  }

  /* =======================================================================
     PLAYOFF
     ======================================================================= */
  /* The 2026 bracket is a fixed shape: seeds 5-12 play the first round at the higher
     seed, 1-4 wait, and every later slot is fed by earlier ones. Building it from a
     seed list rather than reading it out of the export is what lets the scenario
     builder draw a bracket for a field the simulation never produced. `feeds` says
     which earlier games supply each open slot, and matches the map simulate_playoff
     exports for the modal bracket - the two have to agree or the same twelve teams
     would meet in a different order on two tabs. */
  const BRACKET_FEEDS = { 4: [0], 5: [1], 6: [2], 7: [3], 8: [4, 5], 9: [7, 6], 10: [8, 9] };
  function bracketFromSeeds(seeds) {
    const s = i => seeds[i] || null;
    const gm = (round, top, bottom, site) => ({ round, top, bottom, site: site || null });
    return {
      seeds,
      feeds: BRACKET_FEEDS,
      games: [
        gm("R1", s(7), s(8), s(7)), gm("R1", s(4), s(11), s(4)),
        gm("R1", s(5), s(10), s(5)), gm("R1", s(6), s(9), s(6)),
        gm("QF", s(0), null), gm("QF", s(3), null),
        gm("QF", s(2), null), gm("QF", s(1), null),
        gm("SF", null, null), gm("SF", null, null), gm("F", null, null),
      ],
    };
  }

  /* Resolve forward: score each game, advance the projected winner into the slot it
     feeds. Returns a copy - the caller's game list is not mutated. */
  function resolveBracket(games, feeds) {
    const G = games.map(g => ({ ...g }));
    for (let i = 0; i < G.length; i++) {
      const g = G[i];
      const src = (feeds || {})[String(i)] || (feeds || {})[i];
      if (src) {
        const won = src.map(k => G[k].winner).filter(Boolean);
        if (!g.top) g.top = won[0];
        if (!g.bottom) g.bottom = won.length > 1 ? won[1] : (g.top !== won[0] ? won[0] : null);
      }
      if (!g.top || !g.bottom) continue;
      const venue = g.site === g.top ? "A" : g.site === g.bottom ? "B" : "N";
      const r = predict(g.top, g.bottom, venue);
      if (!r) continue;
      g.p = r.pA;
      [g.sa, g.sb] = displayScore(r);
      g.winner = r.pA >= 0.5 ? g.top : g.bottom;
    }
    return G;
  }

  function bracketHTML(G, seeds, trophyNote) {
    const seedOf = {};
    (seeds || []).forEach((t, i) => { if (t) seedOf[t] = i + 1; });
    const side = (t, won, score) => {
      if (!t) return `<div class="slot empty"><span class="seed">–</span>
        <span class="tbd">TBD</span></div>`;
      return `<div class="slot ${won ? "won" : "lost"}"
          style="--wash:${rgba(t, won ? .26 : .06)};--tint:${color(t)}">
        <span class="seed">${seedOf[t] || ""}</span>
        <img src="${logoURL(t)}" alt="" loading="lazy">
        <button class="team-link" data-team="${esc(t)}">${esc(abbr(t))}</button>
        <span class="gscore">${score != null ? score : ""}</span></div>`;
    };
    const card = g => `<div class="game-card">
        ${side(g.top, g.winner && g.winner === g.top, g.sa)}
        ${side(g.bottom, g.winner && g.winner === g.bottom, g.sb)}
        <div class="gmeta">
          <span class="bye-tag">${g.site ? "at " + esc(abbr(g.site)) : "neutral"}</span>
          ${g.p != null ? `<span class="gwp">${pct(Math.max(g.p, 1 - g.p), 0)}</span>` : ""}
        </div></div>`;
    const champ = G[10] && G[10].winner;
    return `
      <div class="round-label">First Round</div><div class="round-label">Quarterfinals</div>
      <div class="round-label">Semifinals</div><div class="round-label">Championship</div>
      <div class="round-col">${G.slice(0, 4).map(card).join("")}</div>
      <div class="round-col">${G.slice(4, 8).map(card).join("")}</div>
      <div class="round-col">${G.slice(8, 10).map(card).join("")}</div>
      <div class="round-col">${card(G[10])}
        ${champ ? `<div class="trophy-card" style="--wash:${rgba(champ, .18)}">
          <div class="emoji">🏆</div><img src="${logoURL(champ)}" alt="">
          <div class="champ-name" style="color:${color(champ)}">${esc(champ)}</div>
          <div class="bye-tag">${trophyNote(champ)}</div></div>` : ""}
      </div>`;
  }

  function renderPlayoff() {
    const playoff = cur().playoff;
    document.getElementById("playoff-meta").innerHTML =
      `Monte Carlo over the full 2026 schedule — <b>${playoff.n_sims.toLocaleString()}
       simulations</b>. Rules: ${esc(playoff.rules)}. Committee ranking modelled on
       every published CFP committee ranking since 2014 (${esc(playoff.committee_proxy || "")}).
`;

    const P = playoff.teams;
    const byTeam = Object.fromEntries(P.map(t => [t.team, t]));
    const br = playoff.bracket;

    if (!br) {
      document.getElementById("bracket").innerHTML =
        `<p class="sub">No bracket in this export — re-run <code>scripts.simulate_playoff</code>.</p>`;
    } else {
      const G = resolveBracket(br.games, br.feeds);
      document.getElementById("bracket").innerHTML = bracketHTML(G, br.seeds,
        champ => `projected champion · ${pct((byTeam[champ] || {}).champ || 0, 1)} title odds`);
    }

    renderSelection(P, byTeam, br);
    renderResumes(P, byTeam, br);
    wireTeamLinks();
  }

  /* ---------- how each team in the bracket got there ----------
     The odds table said a team makes the field 62% of the time. It never said whether
     that team gets in by winning its league or by finishing 9th with two losses, what
     record it needs, or which part of its résumé the committee stand-in is actually
     rewarding. All of that is in the simulation already - it is what the ranking is
     computed FROM - and this shows it.

     Every number here is conditioned on the sims where the team was SELECTED, so it
     describes the average successful season rather than the average season. */
  /* [key, label, fill, ink]. Two colours per term, not one: the fill is chosen to be
     told apart from its neighbours in a 9px bar, and at 11px on a tinted card wash
     those same hues are too pale to read as text. The ink is the same hue darkened
     until it carries on the lightest card. */
  const PART_META = [
    ["win_pct", "Record",   "#3fb950", "#2f7a2c"],
    ["sos",     "Schedule", "#58a6ff", "#2f6096"],
    ["p4",      "League",   "#a371f7", "#6f3fc4"],
    ["rating",  "Quality",  "#d29922", "#8a6113"],
    ["h2h",     "Head&#8209;to&#8209;head", "#f778ba", "#b83a76"],
  ];
  /* Negative terms are clamped away rather than drawn: a stacked bar cannot render a
     negative slice without lying about the widths around it. Head-to-head is the only
     term that goes negative, and a team that made the field on a losing head-to-head
     record is showing that in the number below the bar, which is not clamped. */
  const partsSum = p => PART_META.reduce((s, [k]) => s + Math.max(0, p[k] || 0), 0);

  /* The bar alone said which term was biggest; it never said by how much, and the
     committee score it adds up to - the whole basis of the ranking - was nowhere on
     the page. These are the labels: one figure per term, in that term's colour, and
     the total they sum to. */
  function partsLabels(p) {
    const nums = PART_META.map(([k, lab, , ink]) => {
      const v = p[k] || 0;
      return `<span style="color:${ink}" title="${lab.replace(/&#8209;/g, "-")}">${
        v.toFixed(2)}</span>`;
    }).join("");
    return `<div class="res-nums">${nums}
      <b title="Committee score — the sum of the five terms">${
        PART_META.reduce((s, [k]) => s + (p[k] || 0), 0).toFixed(2)}</b></div>`;
  }

  function resumeCard(t, seed) {
    const p = t.score_parts;
    if (!p) return "";
    const total = partsSum(p) || 1;
    const bar = PART_META.map(([k, , c]) => {
      const v = Math.max(0, p[k] || 0);
      return v <= 0 ? "" :
        `<i style="width:${100 * v / total}%;background:${c}" title="${k}"></i>`;
    }).join("");
    const bid = t.bid || {};
    // A team's route in is whichever of the three it most often takes. Shown as the
    // dominant one plus its share, because "83% as an at-large" says more than three
    // percentages the reader has to rank themselves.
    const routes = [["p4", "Power&nbsp;4 champion"], ["g6", "Group&nbsp;of&nbsp;6 bid"],
                    ["at_large", "at&#8209;large"]]
      .map(([k, lab]) => [bid[k] || 0, lab]).sort((a, b) => b[0] - a[0]);
    const [topShare, topLabel] = routes[0];
    return `<div class="res-card" style="--tint:${color(t.team)};--wash:${rgba(t.team, .10)}">
      <div class="res-head">
        <span class="res-seed">${seed}</span>
        <img src="${logoURL(t.team)}" alt="" loading="lazy">
        <div class="res-name">
          <button class="team-link" data-team="${esc(t.team)}">${esc(t.team)}</button>
          <div class="conf">${esc(t.conference)}</div>
        </div>
        <div class="res-odds"><b>${pct(t.playoff, 0)}</b><span>to make it</span></div>
      </div>
      <div class="res-route">In as <b>${topLabel}</b> ${pct(topShare, 0)} of the time</div>
      <div class="res-bar">${bar}</div>
      ${partsLabels(p)}
      <div class="res-rec">
        <span>Gets in at <b>${t.wins_in != null ? t.wins_in.toFixed(1) : "—"}</b> wins</span>
        <span>Misses at <b>${t.wins_out != null ? t.wins_out.toFixed(1) : "—"}</b></span>
      </div>
    </div>`;
  }

  function renderResumes(P, byTeam, br) {
    const el = document.getElementById("playoff-resumes");
    if (!el) return;
    const seeds = (br && br.seeds) || [];
    const cards = seeds.map((t, i) => (byTeam[t] ? resumeCard(byTeam[t], i + 1) : ""))
                       .join("");

    const legend = PART_META.map(([, lab, c]) =>
      `<span class="res-key"><i style="background:${c}"></i>${lab}</span>`).join("");

    // The bubble. Exactly one team is the last one in and exactly one is the first one
    // out in every simulated season, so these two columns each sum to 100% across the
    // league and are directly comparable.
    const li = P.filter(t => (t.last_in || 0) > 0)
                .sort((a, b) => b.last_in - a.last_in).slice(0, 6);
    const fo = P.filter(t => (t.first_out || 0) > 0)
                .sort((a, b) => b.first_out - a.first_out).slice(0, 6);
    const strip = (list, key, max) => list.map(t => `<div class="ccg-row">
        <img src="${logoURL(t.team)}" alt="" loading="lazy">
        <button class="team-link" data-team="${esc(t.team)}">${esc(abbr(t.team))}</button>
        <div class="bar"><i style="width:${100 * (t[key] || 0) / max}%;
          background:${color(t.team)}"></i></div>
        <span class="pct">${pct(t[key] || 0, 0)}</span></div>`).join("");
    const liMax = Math.max(...li.map(t => t.last_in), 0.01);
    const foMax = Math.max(...fo.map(t => t.first_out), 0.01);

    el.innerHTML = `
      <div class="res-legend">Each bar splits that team's committee score into the
        terms that produced it: ${legend}</div>
      <div class="res-legend sub-note">The row of figures under each bar is those
        same terms as ranking points, in the same order and colours, and the bold
        figure on the right is the committee score they sum to.</div>
      <div class="res-grid">${cards}</div>
      <div class="bubble-wrap">
        <div class="ccg-card"><div class="ccg-name">Last team in</div>${strip(li, "last_in", liMax)}</div>
        <div class="ccg-card"><div class="ccg-name">First team out</div>${strip(fo, "first_out", foMax)}</div>
      </div>
      <div class="wd-foot">The twelfth seed and the best team left behind, as a share
        of all ${cur().playoff.n_sims.toLocaleString()} simulated seasons. Each column
        sums to 100%: every season has exactly one of each.</div>`;
  }

  /* =======================================================================
     SCENARIO BUILDER — pick the games, get the bracket
     =======================================================================
     The Playoff tab answers "how often does this happen", averaged over 20,000
     seasons. This answers the other question people actually ask, which no average
     can: if THESE games go THIS way, who is in? So it plays exactly one season.

     What it is not: odds. Every unpicked game is resolved to the model's favourite,
     which is a season in which no favourite ever loses - not a likely season, just
     the reference one to deviate from. Nothing here feeds the percentages anywhere
     else on the site, and nothing here is a re-run of the Monte Carlo.

     What it IS is the same selection procedure the simulation runs, ported: the same
     committee weights, the same conference-title rule, the same automatic bids, the
     same straight seeding. See scripts/simulate_playoff.py, the per-sim loop.

     The port was checked against that Python by replaying the no-picks season on both
     sides: all twelve seeds, every record, every committee score to two decimals and
     the first team out agree. Anything that changes the arithmetic below should be
     re-checked the same way, because a selection rule that quietly disagrees with the
     simulation would put a different bracket on two tabs of the same site. */
  const SC_STATE = (function () {
    const KEY = "cfb-scenario-v1";
    const games = new Map();      // game key -> winning team
    const titles = new Map();     // conference -> champion
    let ver = 0;
    function save() {
      try {
        localStorage.setItem(KEY, JSON.stringify(
          { games: [...games], titles: [...titles] }));
      } catch (e) { /* private mode; the scenario just will not outlive the tab */ }
    }
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) {
        const j = JSON.parse(raw);
        for (const [k, v] of j.games || []) games.set(k, v);
        for (const [k, v] of j.titles || []) titles.set(k, v);
      }
    } catch (e) { games.clear(); titles.clear(); }
    return {
      version: () => ver,
      count: () => games.size + titles.size,
      game: k => games.get(k),
      title: c => titles.get(c),
      setGame(k, team) { team == null ? games.delete(k) : games.set(k, team); ver++; save(); },
      setTitle(c, team) { team == null ? titles.delete(c) : titles.set(c, team); ver++; save(); },
      reset() { games.clear(); titles.clear(); ver++; save(); },
    };
  })();

  /* Home, away and date. The array index would be shorter but it is only stable until
     the schedule is re-exported, and these picks are kept in localStorage across runs.
     The separator is a pipe rather than the NUL this file uses elsewhere because this
     key makes a round trip through an HTML data- attribute, and the HTML parser
     rewrites NUL to U+FFFD - the key read back off the element then matched nothing
     and every pick was silently ignored. No team name or ISO date contains a pipe. */
  const scKey = g => `${g.h}|${g.a}|${g.d || g.w}`;

  const scPopZ = obj => {                     // ddof=0, as the simulation's .std()
    const ts = Object.keys(obj), v = ts.map(t => obj[t]);
    const mu = v.reduce((a, b) => a + b, 0) / (v.length || 1);
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) * (b - mu), 0) / (v.length || 1));
    return Object.fromEntries(ts.map((t, i) => [t, sd ? (v[i] - mu) / sd : 0]));
  };

  /* The model's favourite for one scheduled game, and the probability behind it.
     Non-FBS opponents have no rating, so the buy game resolves to the FBS team at the
     same 95% the simulation charges it. */
  function scGamePredict(g) {
    const kh = !!vecOf(g.h), ka = !!vecOf(g.a);
    if (kh && ka) {
      const r = predict(g.h, g.a, g.n ? "N" : "A");
      if (!r) return null;
      return { fav: r.pA >= 0.5 ? g.h : g.a, pHome: r.pA, both: true };
    }
    if (!kh && !ka) return null;
    return { fav: kh ? g.h : g.a, pHome: kh ? FCS_WIN_P : 1 - FCS_WIN_P, both: false };
  }

  let scCache = null, scCacheVer = "";
  function scenario() {
    const stamp = SC_STATE.version() + "|" + WI.version();
    if (scCacheVer === stamp && scCache) return scCache;

    const pl = cur().playoff;
    const CW = pl.committee_weights || {};
    // Head-to-head is only defined against a provisional order, and that order comes
    // from the same model fitted WITHOUT the h2h column. Older exports predate that
    // field; falling back to the scoring weights gives a very close but not identical
    // provisional ranking, so the fallback is flagged rather than hidden.
    const PW = pl.provisional_weights || null;
    const P0 = PW || CW;
    const WITHIN = pl.h2h_within || 15;
    const FCS_R = pl.fcs_opp_rating != null ? pl.fcs_opp_rating : -2.0;
    const P4C = new Set(pl.p4_confs || [...P4]);
    const G6C = new Set(pl.g6_confs || [...G6]);

    /* The universe is every team the model carries a vector for, NOT the 136 rows in
       ratings.json. Those two sets differ by the FBS newcomers, who get a
       5th-percentile fallback row in the frame and so are absent from the power
       ratings but very much present on the schedule. Ranking over the smaller set
       charged their opponents the non-FBS schedule penalty and shifted both z-scores,
       which moved every committee score by about 0.01 against the Python. Same set,
       same numbers. */
    const names = Object.keys(cur().model.teams).filter(t => vecOf(t));
    const fbs = new Set(names);

    // rating: z-score of (O + D)/2, exactly as the simulation builds it. Read off the
    // model vectors rather than ratings.json so an edited roster moves it here too.
    const rating = scPopZ(Object.fromEntries(
      names.map(t => { const v = vecOf(t); return [t, (v[0] + v[1]) / 2]; })));

    const wins = {}, losses = {}, played = {}, cWins = {}, cGames = {};
    const sosSum = {}, sosN = {};
    for (const t of names) {
      wins[t] = losses[t] = played[t] = cWins[t] = cGames[t] = 0;
      sosSum[t] = sosN[t] = 0;
    }

    // meetings[] feeds head-to-head; it collects conference title games too, because
    // the simulation counts those as meetings.
    const meetings = [];
    const played_games = [];
    for (const g of schedule) {
      const kh = fbs.has(g.h), ka = fbs.has(g.a);
      if (kh && ka) {
        sosSum[g.h] += rating[g.a]; sosN[g.h]++;
        sosSum[g.a] += rating[g.h]; sosN[g.a]++;
      } else if (kh || ka) {
        const t = kh ? g.h : g.a;
        sosSum[t] += FCS_R; sosN[t]++;
      }
      const P = scGamePredict(g);
      if (!P) continue;
      const key = scKey(g);
      const picked = SC_STATE.game(key);
      const winner = picked || P.fav;
      const loser = winner === g.h ? g.a : g.h;
      played_games.push({ g, key, pred: P, winner, picked: picked || null });
      if (kh && ka) {
        wins[winner]++; losses[loser]++; played[g.h]++; played[g.a]++;
        meetings.push([winner, loser]);
        if (g.c && conf(g.h) === conf(g.a)) {
          cGames[g.h]++; cGames[g.a]++;
          cWins[winner]++;
        }
      } else {
        const t = kh ? g.h : g.a;
        played[t]++;
        if (winner === t) wins[t]++; else losses[t]++;
      }
    }
    const sos = {};
    for (const t of names) sos[t] = sosSum[t] / Math.max(sosN[t], 1);
    const sosZ = scPopZ(sos);

    // --- conference title games: top two by conference win pct, rating as tiebreak
    const champs = {}, titleGames = [];
    for (const c of [...P4C, ...G6C].sort()) {
      const members = names.filter(t => conf(t) === c);
      if (members.length < 2) continue;
      const seedVal = t => (cGames[t] ? cWins[t] / cGames[t] : 0) + 1e-4 * rating[t];
      const [t1, t2] = members.slice().sort((a, b) => seedVal(b) - seedVal(a));
      const r = predict(t1, t2, "N");
      const fav = !r || r.pA >= 0.5 ? t1 : t2;
      const picked = SC_STATE.title(c);
      const winner = picked === t1 || picked === t2 ? picked : fav;
      const loser = winner === t1 ? t2 : t1;
      champs[c] = winner;
      wins[winner]++; played[t1]++; played[t2]++;
      losses[loser]++;
      meetings.push([winner, loser]);
      titleGames.push({ conf: c, t1, t2, winner, fav,
                        pTop: r ? r.pA : 0.5, picked: picked || null });
    }

    const wpct = {};
    for (const t of names) wpct[t] = played[t] ? wins[t] / played[t] : 0;
    // Notre Dame carries the power flag despite being independent — it has a CFP
    // contract slot. Matches scripts/fit_committee.POWER_INDEPENDENTS.
    const isP4 = t => (P4C.has(conf(t)) || t === "Notre Dame") ? 1 : 0;

    const score0 = {};
    for (const t of names) {
      score0[t] = P0.win_pct * wpct[t] + P0.rating * rating[t] +
                  P0.sos * sosZ[t] + (P0.power_conf || 0) * isP4(t);
    }
    const rank0 = {};
    names.slice().sort((a, b) => score0[b] - score0[a])
         .forEach((t, i) => { rank0[t] = i; });

    const h2h = {};
    for (const t of names) h2h[t] = 0;
    if (CW.h2h) {
      for (const [w, l] of meetings) {
        if (Math.abs(rank0[w] - rank0[l]) <= WITHIN) { h2h[w]++; h2h[l]--; }
      }
    }

    const parts = {}, score = {};
    for (const t of names) {
      parts[t] = {
        win_pct: CW.win_pct * wpct[t],
        rating: CW.rating * rating[t],
        sos: CW.sos * sosZ[t],
        p4: (CW.power_conf || 0) * isP4(t),
        h2h: (CW.h2h || 0) * h2h[t],
      };
      score[t] = parts[t].win_pct + parts[t].rating + parts[t].sos +
                 parts[t].p4 + parts[t].h2h;
    }

    // --- selection: 4 P4 champions + best G6 + at-larges, straight seeding
    const autos = [...P4C].sort().map(c => champs[c]).filter(Boolean);
    const g6pool = names.filter(t => G6C.has(conf(t)));
    const g6bid = g6pool.length
      ? g6pool.reduce((best, t) => score[t] > score[best] ? t : best, g6pool[0])
      : null;
    if (g6bid) autos.push(g6bid);
    const field = new Set(autos);
    const order = names.slice().sort((a, b) => score[b] - score[a]);
    for (const t of order) {
      if (field.size >= 12) break;
      field.add(t);
    }
    const seeds = [...field].sort((a, b) => score[b] - score[a]).slice(0, 12);
    const bubble = order.find(t => !field.has(t)) || null;

    const route = {};
    for (const t of seeds) {
      const c = Object.keys(champs).find(k => champs[k] === t && P4C.has(k));
      route[t] = c ? `${c} champion` : (t === g6bid ? "Group of 6 bid" : "at-large");
    }

    const br = bracketFromSeeds(seeds);
    scCache = {
      names, wins, losses, played, cWins, cGames, wpct, rating, sosZ, h2h,
      score, parts, champs, titleGames, seeds, bubble, route, g6bid,
      games: played_games, bracket: br, resolved: resolveBracket(br.games, br.feeds),
      provisional: !!PW,
    };
    scCacheVer = stamp;
    return scCache;
  }

  /* ---------- scenario rendering ---------- */
  const SC_LIMIT = 250;
  let scMode = "pick";                 // "pick" | "playoff"
  function scFilteredGames(S) {
    const q = document.getElementById("sc-search").value.trim().toLowerCase();
    const c = document.getElementById("sc-conf").value;
    const wk = document.getElementById("sc-week").value;
    const sc = document.getElementById("sc-scope").value;
    let rows = S.games;
    if (q) rows = rows.filter(r => r.g.h.toLowerCase().includes(q) ||
                                   r.g.a.toLowerCase().includes(q));
    if (c) rows = rows.filter(r => conf(r.g.h) === c || conf(r.g.a) === c);
    if (wk) rows = rows.filter(r => String(r.g.w) === wk);
    if (sc === "picked") rows = rows.filter(r => r.picked);
    else if (sc === "close") rows = rows.filter(r =>
      r.pred.both && Math.abs(r.pred.pHome - 0.5) <= 0.10);
    else if (sc === "conf") rows = rows.filter(r =>
      r.g.c && conf(r.g.h) === conf(r.g.a));
    return rows;
  }

  function scChip(team, p, on, picked, key, side) {
    const tint = color(team);
    return `<button class="sc-chip${on ? " on" : ""}${picked ? " picked" : ""}"
      style="--tint:${tint};--wash:${rgba(team, on ? .22 : .05)}"
      data-key="${esc(key)}" data-team="${esc(team)}" data-side="${side}"
      title="${esc(team)} — model gives ${pct(p, 0)}">
      <img src="${logoURL(team)}" alt="" loading="lazy">
      <span class="sc-abbr">${esc(abbr(team))}</span>
      <span class="sc-p">${pct(p, 0)}</span></button>`;
  }

  function renderScenario() {
    const host = document.getElementById("sc-body");
    if (!host) return;
    const S = scenario();

    document.getElementById("sc-reset").disabled = !SC_STATE.count();
    const base = new Set((cur().playoff.bracket || {}).seeds || []);
    const added = S.seeds.filter(t => !base.has(t));
    const dropped = [...base].filter(t => !S.seeds.includes(t));

    const rows = scFilteredGames(S);
    const shown = rows.slice(0, SC_LIMIT);
    const picked = SC_STATE.count() ? `${SC_STATE.count()} picked` : "nothing picked";
    document.getElementById("sc-count").textContent = scMode === "playoff" ? picked
      : `${rows.length > SC_LIMIT ? `first ${SC_LIMIT} of ` : ""}${rows.length} games · ${picked}`;

    const gameRows = shown.map(r => {
      const { g, key, pred } = r;
      const pickedH = r.picked === g.h, pickedA = r.picked === g.a;
      return `<div class="sc-game${r.picked ? " has-pick" : ""}">
        <span class="sc-wk">Wk ${g.w}</span>
        <span class="sc-date">${g.d ? g.d.slice(5) : ""}</span>
        ${scChip(g.a, 1 - pred.pHome, r.winner === g.a, pickedA, key, "a")}
        <span class="sc-at">${g.n ? "vs" : "at"}</span>
        ${scChip(g.h, pred.pHome, r.winner === g.h, pickedH, key, "h")}
        ${r.picked ? `<button class="sc-undo" data-key="${esc(key)}"
          title="Back to the model's pick">undo</button>` : `<span class="sc-undo-sp"></span>`}
      </div>`;
    }).join("");

    const titleRows = S.titleGames.map(t => `<div class="sc-game sc-ccg${
      t.picked ? " has-pick" : ""}">
        <span class="sc-wk sc-ccg-name">${esc(t.conf)}</span>
        ${scChip(t.t1, t.pTop, t.winner === t.t1, t.picked === t.t1, "ccg:" + t.conf, "1")}
        <span class="sc-at">vs</span>
        ${scChip(t.t2, 1 - t.pTop, t.winner === t.t2, t.picked === t.t2, "ccg:" + t.conf, "2")}
        ${t.picked ? `<button class="sc-undo" data-key="ccg:${esc(t.conf)}"
          title="Back to the model's pick">undo</button>` : `<span class="sc-undo-sp"></span>`}
      </div>`).join("");

    const seedRows = S.seeds.map((t, i) => `<div class="sc-seed">
        <span class="sc-seed-n${i < 4 ? " bye" : ""}">${i + 1}</span>
        <img src="${logoURL(t)}" alt="" loading="lazy">
        <div class="sc-seed-id">
          <button class="team-link" data-team="${esc(t)}">${esc(t)}</button>
          <div class="conf">${esc(S.route[t])}</div>
        </div>
        <span class="sc-rec">${S.wins[t]}–${S.losses[t]}</span>
        <span class="sc-score" title="Committee score">${S.score[t].toFixed(2)}</span>
        ${added.includes(t) ? `<span class="tag stars">NEW</span>` : ""}
      </div>`).join("");

    const diff = (!added.length && !dropped.length)
      ? `Same twelve as the projected field.`
      : `${added.length ? `<b>In:</b> ${added.map(esc).join(", ")}. ` : ""}${
          dropped.length ? `<b>Out:</b> ${dropped.map(esc).join(", ")}.` : ""}`;
    const champ = S.resolved[10] && S.resolved[10].winner;

    /* Two modes rather than one long page. Picking games and reading the bracket are
       different activities: the schedule wants a tall scrolling list, the bracket
       wants width, and putting them side by side gave the bracket a quarter of the
       screen while the list you were actually working in sat below the fold.

       The cost of splitting them is that a pick stops being visibly consequential, so
       pick mode keeps a one-line readout of what the picks have produced - champion,
       and what changed about the field - with the way through to the full bracket. */
    const summary = `<div class="sc-summary">
      <span class="sc-sum-champ">${champ
        ? `<img src="${logoURL(champ)}" alt="" loading="lazy"><b>${esc(champ)}</b>
           <span>wins it</span>` : "—"}</span>
      <span class="sc-sum-diff">${diff}</span>
      <button class="sc-goto" data-mode="playoff">See the playoff &rarr;</button>
    </div>`;

    const playoffHTML = `
      <div class="sc-out">
        <div class="panel sc-field">
          <h3>The field <span class="hint">— committee order, straight seeding</span></h3>
          ${seedRows}
          <div class="sc-bubble">First team out: ${S.bubble
            ? `<b>${esc(S.bubble)}</b> (${S.wins[S.bubble]}–${S.losses[S.bubble]},
               ${S.score[S.bubble].toFixed(2)})` : "—"}</div>
          <div class="sc-diff">${diff}</div>
        </div>
        <div class="panel sc-bracket">
          <h3>The bracket</h3>
          <div id="sc-bracket-grid">${bracketHTML(S.resolved, S.seeds,
            () => "wins this scenario")}</div>
        </div>
      </div>
      <div class="sc-back-row">
        <button class="sc-goto" data-mode="pick">&larr; Back to the games</button>
        ${SC_STATE.count() ? "" : `<span class="hint">Nothing picked yet &mdash; this
          is the season in which every favourite wins.</span>`}
      </div>`;

    const pickHTML = `
      ${summary}
      <h3 class="bracket-title">Conference title games
        <span class="hint">— the top two in each league by conference record; pick the
          winner and the four Power 4 champions change with it</span></h3>
      <div class="sc-games sc-ccg-list">${titleRows}</div>

      <h3 class="bracket-title">The schedule
        <span class="hint">— click a team to make it the winner; everything unpicked
          stays on the model's favourite</span></h3>
      <div class="sc-games">${gameRows || `<div class="wd-foot">No games match
        these filters.</div>`}</div>
      ${rows.length > SC_LIMIT ? `<div class="wd-foot">Showing the first ${SC_LIMIT}
        games in schedule order. Narrow with the filters above to reach the rest &mdash;
        picks you have already made are kept whether or not they are on screen.</div>` : ""}`;

    host.innerHTML = (scMode === "playoff" ? playoffHTML : pickHTML) + `
      <div class="wd-foot">One season, not twenty thousand: every game you have not
        picked is resolved to the model's favourite, so this is the chalk season with
        your results substituted in, and the field is an outcome rather than a
        probability. Selection follows the real rule &mdash; four Power&nbsp;4
        champions, the highest-ranked Group of 6 team, seven at-large, straight
        seeding &mdash; and ranks teams with the same fitted committee weights the
        simulation uses.${S.provisional ? "" : ` <b>Note:</b> this export predates the
        head-to-head provisional weights, so near-ties may break slightly differently
        from the Playoff tab.`}</div>`;

    host.querySelectorAll(".sc-goto").forEach(b =>
      b.addEventListener("click", () => scSetMode(b.dataset.mode)));
    host.querySelectorAll(".sc-chip").forEach(b =>
      b.addEventListener("click", () => {
        const key = b.dataset.key, team = b.dataset.team;
        if (key.startsWith("ccg:")) {
          const c = key.slice(4);
          SC_STATE.setTitle(c, SC_STATE.title(c) === team ? null : team);
        } else {
          SC_STATE.setGame(key, SC_STATE.game(key) === team ? null : team);
        }
        scRerender();
      }));
    host.querySelectorAll(".sc-undo").forEach(b =>
      b.addEventListener("click", () => {
        const key = b.dataset.key;
        if (key.startsWith("ccg:")) SC_STATE.setTitle(key.slice(4), null);
        else SC_STATE.setGame(key, null);
        scRerender();
      }));
    wireTeamLinks();
  }

  // The list can be 250 rows tall, and re-rendering it drops the page wherever the
  // browser feels like. Hold the scroll position across a pick.
  function scRerender() {
    const y = window.scrollY;
    renderScenario();
    window.scrollTo(0, y);
  }

  /* Which half of the tab is showing. Switching goes back to the top rather than
     holding scroll: the two modes are different lengths, and keeping a schedule-list
     scroll position on a one-screen bracket lands the reader past the end of it. */
  function scSetMode(mode) {
    scMode = mode === "playoff" ? "playoff" : "pick";
    document.querySelectorAll("#sc-mode .seg-btn").forEach(b =>
      b.classList.toggle("active", b.dataset.mode === scMode));
    // The filters only drive the schedule list, so they go with it.
    const bar = document.getElementById("sc-toolbar");
    if (bar) bar.classList.toggle("sc-showing-playoff", scMode === "playoff");
    renderScenario();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  document.querySelectorAll("#sc-mode .seg-btn").forEach(b =>
    b.addEventListener("click", () => scSetMode(b.dataset.mode)));

  function fillScenarioSelects() {
    const cs = document.getElementById("sc-conf");
    if (cs && cs.options.length <= 1) {
      [...new Set(cur().ratings.teams.map(t => t.conference))].sort()
        .forEach(c => cs.add(new Option(c, c)));
    }
    const ws = document.getElementById("sc-week");
    if (ws && ws.options.length <= 1) {
      [...new Set(schedule.map(g => g.w))].filter(w => w != null)
        .sort((a, b) => a - b).forEach(w => ws.add(new Option("Week " + w, w)));
    }
  }

  ["sc-search", "sc-conf", "sc-week", "sc-scope"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", renderScenario);
  });
  const scReset = document.getElementById("sc-reset");
  if (scReset) scReset.addEventListener("click", () => { SC_STATE.reset(); renderScenario(); });

  const P4_CONFS = ["ACC", "Big 12", "Big Ten", "SEC"];
  const G6_CONFS = ["American Athletic", "Conference USA", "Mid-American",
                    "Mountain West", "Pac-12", "Sun Belt"];

  /* How the twelve teams are chosen, and who is in line for the five automatic bids.
     The bracket above shows an outcome; this shows the rule that produced it. Worth
     spelling out because the 2026 format changed - the Group of 6 bid is no longer
     tied to winning a conference - and because the ranking underneath is a fitted
     model of the committee rather than the model's own opinion of who is best. */
  function renderSelection(P, byTeam, br) {
    const el = document.getElementById("playoff-explain");
    const pl = cur().playoff;
    const cw = pl.committee_weights;

    /* Step 2 asserted an ordering - record, then schedule, then quality - and asked to
       be taken on faith. This is that claim as a measurement: the average selected
       team's committee score, term by term, over the twelve in the bracket above. The
       raw fitted weights cannot be compared to each other (win pct is a fraction, the
       other inputs are z-scores and flags), so what is charted is the CONTRIBUTION each
       term actually makes, which is the thing the sentence is about. Same colours as
       the résumé cards below, so a reader can carry the key down the page. */
    const fieldTeams = ((br && br.seeds) || []).map(t => byTeam[t])
      .filter(t => t && t.score_parts);
    let weightsHTML = "";
    if (fieldTeams.length) {
      const avg = {};
      for (const [k] of PART_META) {
        avg[k] = fieldTeams.reduce((s, t) => s + (t.score_parts[k] || 0), 0) /
                 fieldTeams.length;
      }
      const span = Math.max(...PART_META.map(([k]) => Math.abs(avg[k]))) || 1;
      const rows = PART_META.map(([k, lab, c, ink]) => `<div class="cw-row">
          <span class="cw-lab">${lab}</span>
          <div class="cw-track"><i style="width:${100 * Math.abs(avg[k]) / span}%;
            background:${c}"></i></div>
          <span class="cw-val" style="color:${ink}">${avg[k].toFixed(2)}</span>
        </div>`).join("");
      weightsHTML = `<div class="cw-chart">
        <div class="cw-head">What the ranking is actually made of
          <span class="hint">— average ranking points per term across the twelve teams
            in the projected field</span></div>
        ${rows}</div>`;
    }

    if (el) el.innerHTML = `
      <div class="steps">
        <div class="step"><span class="step-n">1</span>
          <h4>Play the season</h4>
          <p>All ${schedule.length} scheduled games are simulated
            ${pl.n_sims.toLocaleString()} times, then each conference stages a title
            game between its top two finishers.</p></div>
        <div class="step"><span class="step-n">2</span>
          <h4>Rank the teams</h4>
          <p>A stand-in for the selection committee sorts everyone. It is fitted to
            every ranking the real committee has published since 2014, and what it
            learned is that the committee cares most about <b>record</b>, then
            <b>schedule strength</b>, and much less about how good a team looks. It
            also breaks near-ties on <b>head-to-head</b>: beating a team ranked close
            to you counts, and losing to one costs.</p>
          ${weightsHTML}</div>
        <div class="step"><span class="step-n">3</span>
          <h4>Hand out 5 automatic bids</h4>
          <p>The champions of the ACC, Big&nbsp;12, Big&nbsp;Ten and SEC are in no
            matter where they are ranked, plus the highest-ranked team from the other
            six conferences &mdash; <b>new for 2026</b>, that team no longer has to win
            its league.</p></div>
        <div class="step"><span class="step-n">4</span>
          <h4>Fill the last 7 and seed</h4>
          <p>The seven highest-ranked teams left over get at-large bids. Seeding is
            straight down the ranking, 1 through 12: the top four sit out the first
            round, and seeds 5&ndash;8 host it.</p></div>
      </div>
      <div class="wd-foot">The committee stand-in scores each team
        <b>${cw ? cw.win_pct.toFixed(1) : "?"} &times; win&nbsp;pct
        + ${cw ? cw.sos.toFixed(2) : "?"} &times; schedule strength
        + ${cw ? cw.rating.toFixed(2) : "?"} &times; team rating</b>. Those weights are
        fitted, not chosen &mdash; and the fact that record and schedule dwarf team
        quality is what the historical rankings say, not an assumption. Leaving one
        season out at a time, it reproduces the real final ranking at a rank
        correlation of 0.91.</div>`;

    const ccg = document.getElementById("playoff-ccg");
    if (!ccg) return;
    const block = (title, confs, auto) => `
      <h4 class="ccg-h">${title}</h4>
      <div class="ccg-grid">${confs.map(c => {
        const pool = P.filter(t => t.conference === c)
                      .sort((a, b) => b.conf_champ - a.conf_champ).slice(0, 4);
        if (!pool.length) return "";
        return `<div class="ccg-card">
          <div class="ccg-name">${esc(c)}
            <span class="ccg-tag">${auto ? "champion is in" : "no auto bid"}</span></div>
          ${pool.map(t => `<div class="ccg-row">
            <img src="${logoURL(t.team)}" alt="" loading="lazy">
            <button class="team-link" data-team="${esc(t.team)}">${esc(abbr(t.team))}</button>
            <div class="bar"><i style="width:${100 * t.conf_champ}%;
              background:${color(t.team)}"></i></div>
            <span class="pct">${pct(t.conf_champ, 0)}</span></div>`).join("")}
        </div>`;
      }).join("")}</div>`;

    const g6all = P.filter(t => G6_CONFS.includes(t.conference))
                   .sort((a, b) => (b.g6_bid || 0) - (a.g6_bid || 0));
    const g6 = g6all.slice(0, 8);
    // The bid rotates across a dozen candidates, so no one team is near 100% and a
    // raw percentage scale leaves every bar a stub. Scale to the leader.
    const g6max = Math.max(...g6.map(t => t.g6_bid || 0), 0.01);
    ccg.innerHTML = block("Power 4 — champion is in automatically", P4_CONFS, true)
      + block("Group of 6 — one bid, and it goes to the ranking, not the trophy",
              G6_CONFS, false)
      + `<h4 class="ccg-h">Who takes the Group of 6 bid</h4>
         <div class="ccg-card wide">${g6.map(t => `<div class="ccg-row">
           <img src="${logoURL(t.team)}" alt="" loading="lazy">
           <button class="team-link" data-team="${esc(t.team)}">${esc(t.team)}</button>
           <div class="bar"><i style="width:${100 * (t.g6_bid || 0) / g6max}%;
             background:${color(t.team)}"></i></div>
           <span class="pct">${pct(t.g6_bid || 0, 0)}</span></div>`).join("")}</div>
         <div class="wd-foot">Share of simulated seasons in which this team is the
           highest-ranked Group of 6 team, and therefore the one that goes.</div>`;
  }

  /* =======================================================================
     MATCHUP SIMULATOR
     ======================================================================= */
  function makePicker(el, initial) {
    const byConf = {};
    rankNames().slice().sort().forEach(t => (byConf[conf(t)] = byConf[conf(t)] || []).push(t));
    const sel = document.createElement("select");
    sel.innerHTML = Object.keys(byConf).sort().map(c =>
      `<optgroup label="${esc(c)}">` +
      byConf[c].map(t => `<option value="${esc(t)}"${t === initial ? " selected" : ""}>${esc(t)}</option>`).join("") +
      `</optgroup>`).join("");
    el.innerHTML = "";
    el.appendChild(sel);
    return sel;
  }
  const selA = makePicker(document.getElementById("pickerA"), "Ohio State");
  const selB = makePicker(document.getElementById("pickerB"), "Notre Dame");
  const selT = makePicker(document.getElementById("pickerTeam"), "Ohio State");

  function renderMatchup() {
    const a = selA.value, b = selB.value;
    document.getElementById("venueA-label").textContent = "At " + abbr(a);
    document.getElementById("venueB-label").textContent = "At " + abbr(b);
    const venue = document.querySelector("input[name=venue]:checked").value;
    if (a === b) {
      document.getElementById("matchup-result").innerHTML =
        `<p class="sub" style="text-align:center">Pick two different teams 🙂</p>`;
      return;
    }
    const r = predict(a, b, venue);
    const fav = r.margin >= 0 ? a : b, spread = Math.abs(r.margin);
    const venueNote = venue === "N" ? "Neutral field" : "At " + (venue === "A" ? a : b);
    const [sa, sb] = displayScore(r);
    // The margins that actually happen, from the fitted distribution rather than the
    // normal the win model uses - it puts .044 on a 3-point game against a real .106.
    // This is a statement about the spread, not a rounding of it, which is why it
    // survives the scoreline snapping that used to sit above it.
    const KEYS = [3, 7, 10, 14];
    const keyP = KEYS.map(k => [k, keyNumberP(r.margin, k)]);
    const keyRow = keyP[0][1] == null ? "" : `
      <div class="keynums">
        <span class="kn-label">Chance the game is decided by exactly</span>
        ${keyP.map(([k, p]) => `<span class="kn"><b>${k}</b>${pct(p, 1)}</span>`).join("")}
      </div>`;
    document.getElementById("matchup-result").innerHTML = `
      <div class="face-off">
        <div class="side">
          <img src="${logoURL(a)}" alt=""><div class="name">${esc(a)}</div>
          <div class="conf">${esc(conf(a))}</div>
          <div class="winp" style="color:${color(a)}">${pct(r.pA)}</div>
        </div>
        <div class="mid">
          <div class="score">${sa} · ${sb}</div>
          <div>projected score</div><div class="venue-note">${esc(venueNote)}</div>
        </div>
        <div class="side">
          <img src="${logoURL(b)}" alt=""><div class="name">${esc(b)}</div>
          <div class="conf">${esc(conf(b))}</div>
          <div class="winp" style="color:${color(b)}">${pct(1 - r.pA)}</div>
        </div>
      </div>
      <div class="prob-strip">
        <div style="width:${100 * r.pA}%;background:${color(a)}"></div>
        <div style="width:${100 * (1 - r.pA)}%;background:${color(b)}"></div>
      </div>
      <div class="stat-row">
        <span><b>${esc(abbr(fav))} −${spread.toFixed(1)}</b> spread</span>
        <span><b>${r.total.toFixed(1)}</b> total (O/U)</span>
        <span><b>${sa}–${sb}</b> projected score</span>
      </div>
      ${keyRow}`;
  }
  selA.addEventListener("change", renderMatchup);
  selB.addEventListener("change", renderMatchup);
  document.querySelectorAll("input[name=venue]").forEach(x =>
    x.addEventListener("change", renderMatchup));

  /* =======================================================================
     TEAM BREAKDOWN
     ======================================================================= */
  const GROUP_ORDER = ["QB", "RB", "WR", "TE", "OT", "IOL",
                       "DT", "EDGE", "LB", "CB", "SAF"];
  const OFF_GROUPS = new Set(["QB", "RB", "WR", "TE", "OT", "IOL"]);

  /* The line is TWO value groups but ONE row of five men on the field. Tackles and
     interior linemen are valued separately - different jobs, separately fitted facet
     weights - so `p.g` is OT or IOL everywhere a number is shown. The formation
     diagram is the one place that has to put them back together, because spreading
     each about the centre independently would stack the guards on top of the tackles.
     layoutGroup() is that seam, and it is used ONLY for placement and the eleven-man
     budget, never for a value. */
  const LINE_GROUPS = new Set(["OT", "IOL"]);
  const layoutGroup = p => (LINE_GROUPS.has(p.g) ? "OL" : p.g);
  const FCS_WIN_P = 0.95;   // matches the simulation's buy-game assumption

  /* ---- formation layout -------------------------------------------------
     The two-deep already carries real positional labels - LT/LG/C/RG/RT, WR-X /
     WR-Z / WR-SL, NB for the nickel - so each team's listed starters ARE its base
     personnel. Nothing needs to be assumed: most teams come out in 11 personnel
     and a four-man front with a nickel back, and the ones that don't lay out as
     whatever they actually list. Coordinates are percentages of the field box,
     offense driving up the screen and defense down it. */
  // Both boxes read the same way: the line of scrimmage sits at y = 46, everything
  // downfield is above it, everything behind the ball is below. Offence therefore
  // fills the lower half and defence the upper half of its own box.
  const LOS = 46;

  /* Placement is driven by position GROUP and unit size, not by the alignment label.
     Ourlads charts carry roughly eighty different labels - the edge rusher alone
     appears as JACK, RUSH, BAN, BUCK, LEO, STUD, VIPER, WOLF, STING, JOKER, CAT, DOG
     and SPEAR - so a lookup table keyed on the label leaves anything unrecognised
     stacked at the centre of the field, which is what scattered the secondary. Each
     level instead spreads symmetrically about the middle, so every unit is centred
     and evenly spaced whatever its personnel and whatever the program calls it. */
  const LEVELS = {
    OL:   { y: 46, gap: 11.5 },
    TE:   { y: 46, flank: "OL", out: 12 },  // outside the tackles, not inside them
    WR:   { y: 44, wide: true },            // on the numbers, slot tucked inside
    QB:   { y: 62, gap: 14 },
    RB:   { y: 78, gap: 16 },
    DT:   { y: 37, gap: 13 },
    EDGE: { y: 35, flank: "DT", out: 14 },  // outside the interior line
    LB:   { y: 24, gap: 17 },
    CB:   { y: 30, wide: true },
    SAF:  { y: 9,  gap: 26 },
  };
  const CENTER = 50;

  /* n items spread symmetrically about `c`. One item lands exactly on centre. */
  function spread(n, gap, c = CENTER) {
    return Array.from({ length: n }, (_, i) => c + (i - (n - 1) / 2) * gap);
  }
  /* Wide groups (receivers, corners) work outside-in: the first two take the
     boundaries, extras tuck inside as slots and nickels. */
  function wideSpread(n) {
    const outer = [7, 93], inner = [21, 79, 31, 69];
    return Array.from({ length: n },
      (_, i) => i < 2 ? outer[i] : (inner[i - 2] ?? CENTER));
  }
  /* Flanking groups line up outside whatever they play beyond. Side comes from the
     label where there is one - a left end belongs on the left - and otherwise the
     first flanker takes the strong side and the next goes opposite. */
  function flankSpread(members, extent, out) {
    const [lo, hi] = extent;
    let nl = 0, nr = 0;
    return members.map(p => {
      const s = sideKey(p);
      const left = s === -1 || (s !== 1 && nr > nl);
      return left ? lo - out - (nl++) * out : hi + out + (nr++) * out;
    });
  }

  /* Preserve the side a label implies so a left corner stays left and a slot or
     nickel sorts to the end, where the inside positions are. */
  function sideKey(p) {
    const k = (p.p || "").toUpperCase();
    if (/SL|SLOT|^NB|STAR|NICKEL|CASH|MONEY|HUSKY|SPUR/.test(k)) return 2;
    if (/^L|^WLB|^WILL|-X$/.test(k)) return -1;
    if (/^R|^SLB|^SAM|-Z$/.test(k)) return 1;
    return 0;
  }

  /* Left-to-right order where the position label actually implies one.
     sideKey resolves only left / middle / right, which is enough for a secondary but
     not for a line: LT and LG both scored -1, so which of them ended up outside came
     down to the order the members happened to be in - and toEleven has already sorted
     them by projected value by then. Ohio State's two-deep lists RG, RT, LG, C, LT and
     rendered with the left guard outside the left tackle.
     QT/QG/SG/ST are the strong-side/quick-side naming some charts use instead of
     left/right; they occupy the same five slots in the same order. */
  const LINE_ORDER = { LT: -2, QT: -2, LG: -1, QG: -1, C: 0, OC: 0,
                       RG: 1, SG: 1, RT: 2, ST: 2 };
  function orderKey(p) {
    const k = (p.p || "").toUpperCase();
    // x2 so the fallback shares a scale with the line ordinals and slot/nickel (2)
    // still sorts last
    return k in LINE_ORDER ? LINE_ORDER[k] : sideKey(p) * 2;
  }

  function shortName(n) {
    const parts = String(n).trim().split(/\s+/);
    return parts.length < 2 ? n : `${parts[0][0]}. ${parts.slice(1).join(" ")}`;
  }

  function placeSide(players, tint, groups) {
    const html = [];
    const extent = {};                    // group -> [leftmost x, rightmost x]
    for (const g of groups) {
      const lvl = LEVELS[g];
      const members = players.filter(p => layoutGroup(p) === g);
      if (!members.length || !lvl) continue;
      // left-ish labels first, slot/nickel last, so sides come out where expected
      members.sort((a, b) => orderKey(a) - orderKey(b));
      const xs = lvl.wide ? wideSpread(members.length)
        : lvl.flank ? flankSpread(members, extent[lvl.flank] || [CENTER, CENTER],
                                  lvl.out)
        : spread(members.length, lvl.gap);
      extent[g] = [Math.min(...xs), Math.max(...xs)];
      members.forEach((p, i) => {
        const x = Math.max(4, Math.min(96, xs[i]));
        // a slot receiver or nickel sits a step off the line rather than on it
        const inside = lvl.wide && i >= 2;
        const y = lvl.y + (inside ? (g === "WR" ? 11 : -13) : 0);
        const label = (p.p || p.g || "").toUpperCase().replace(/^WR-/, "").slice(0, 5);
        html.push(`<div class="plr" style="left:${x}%;top:${y}%"
            title="${esc(p.n)} — ${esc(p.p || p.g)} — ${p.w.toFixed(2)} wins added">
          <div class="dot" style="background:${tint}">${esc(label)}</div>
          <div class="nm">${esc(shortName(p.n))}${p.i
            ? ` <span class="tag unproven" title="No prior FBS snaps — this projection is a positional prior, not a measurement">?</span>` : ""}</div>
          <div class="vl">${p.w >= 0 ? "+" : ""}${p.w.toFixed(2)}</div>
        </div>`);
      });
    }
    return html.join("");
  }

  /* Trim a unit to eleven.

     Ourlads lists the base package, but for 31 of 136 teams that comes to twelve
     because the chart describes two packages at once. Texas is the clearest case:
     three receivers AND a second tight end on offence, three linebackers AND a
     nickel back on defence. Taking every depth-1 player therefore put twelve men on
     the field.

     Trimming is by a fixed convention rather than by value, so the same team always
     produces the same lineup. Offence keeps 11 personnel - a back, a tight end and
     three receivers - and defence keeps the nickel, because five defensive backs is
     the modern base and the third linebacker is the situational one. Anyone trimmed
     is still listed in the contributors table; only the field diagram is capped. */
  // group -> [budget, base rank, rank once the budget is used up]. Budgets add to 11
  // on each side. Ranking everyone and then taking the top eleven means a team that
  // is short in one group back-fills from the next-cheapest surplus on its own - a
  // three-man front ends up with an extra linebacker rather than ten men.
  const ELEVEN = {
    off: { QB: [1, 0, 30], OL: [5, 1, 34], RB: [1, 6, 33],
           WR: [3, 7, 32], TE: [1, 10, 31] },
    def: { DT: [2, 1, 34], EDGE: [2, 3, 34.5], LB: [2, 5, 30],
           CB: [3, 7, 32], SAF: [2, 10, 33] },
  };

  function toEleven(players, isOffense) {
    const budget = isOffense ? ELEVEN.off : ELEVEN.def;
    const seen = {};
    // Order within a group by projected value first, so the man who comes off is the
    // least valuable one rather than whoever the chart happened to list last. Going
    // by list order dropped Florida's middle linebacker and one of Florida Atlantic's
    // outside receivers.
    return players.slice()
      .sort((a, b) => (b.w ?? 0) - (a.w ?? 0))
      .map(p => {
        const g = layoutGroup(p);
        const b = budget[g];
        const n = (seen[g] = (seen[g] || 0) + 1);
        if (!b) return { p, r: 60 + n };
        const [cap, base, over] = b;
        return { p, r: n <= cap ? base + (n - 1) : over + n * 0.01 };
      })
      .sort((a, b) => a.r - b.r)
      .slice(0, 11)
      .map(x => x.p);
  }

  function lineupHTML(team, roster, tint) {
    // The published build ships team and position-group totals but no per-player
    // rows, because those are derived from licensed PFF grades. Everything else on
    // this page still works; only the field diagram needs individual players.
    if (!roster.players) {
      return `<p class="sub">Individual player projections are not included in this
        build. The position-group figures below are the same numbers, summed.</p>`;
    }
    // `st` is the model's starter flag, which is NOT the same as being listed first:
    // the quarterback sheet, the injury file and the position-slot repair all move it
    // off the listed first-teamer. Falling back to depth keeps an older players.json
    // rendering rather than showing an empty field.
    const hasFlag = roster.players.some(p => p.st !== undefined);
    const starters = roster.players.filter(p => hasFlag ? p.st : p.d === 1);
    if (starters.length < 8) {
      return `<p class="sub">No depth chart available for this team.</p>`;
    }
    const offAll = starters.filter(p => OFF_GROUPS.has(p.g));
    const defAll = starters.filter(p => !OFF_GROUPS.has(p.g));
    const off = toEleven(offAll, true);
    const def = toEleven(defAll, false);
    // Say so when a chart is short rather than drawing a gap and leaving it unexplained.
    const fmt = n => n === 11 ? "11 on the field"
      : `${n} on the field — the published chart lists only ${n}`;
    return `<div class="lineup-wrap">
      <div class="field">
        <div class="field-label">Offense · ${esc(personnelLabel(off))}</div>
        <div class="los" style="top:${LOS}%"></div>
        ${placeSide(off, tint, ["OL", "TE", "WR", "QB", "RB"])}
        <div class="field-note">${fmt(off.length)}</div>
      </div>
      <div class="field">
        <div class="field-label">Defense · ${esc(frontLabel(def))}</div>
        <div class="los" style="top:${LOS}%"></div>
        ${placeSide(def, tint, ["DT", "EDGE", "LB", "CB", "SAF"])}
        <div class="field-note">${fmt(def.length)}</div>
      </div>
    </div>`;
  }

  /* Personnel grouping is named by the count of backs and tight ends, the way a
     coach would say it: one back and one tight end is 11, one and two is 12. */
  function personnelLabel(off) {
    const rb = off.filter(p => p.g === "RB").length;
    const te = off.filter(p => p.g === "TE").length;
    return `${rb}${te} personnel`;
  }
  /* The reconciliation the raw WAR number never showed: replacement floor, what the
     roster is worth, what the schedule is worth, and the projected record they add
     up to. Keeping the schedule term visible is the point - it is why a good MAC
     roster and a middling SEC roster can project the same number of wins. */
  function frontLabel(def) {
    const dl = def.filter(p => p.g === "DT" || p.g === "EDGE").length;
    const lb = def.filter(p => p.g === "LB").length;
    const db = def.filter(p => p.g === "CB" || p.g === "SAF").length;
    const base = `${dl}-${lb}`;
    return db >= 5 ? `${base} nickel` : base;
  }

  function teamSchedule(t) {
    const out = [];
    for (const g of schedule) {
      if (g.h !== t && g.a !== t) continue;
      const home = g.h === t, opp = home ? g.a : g.h;
      const known = !!vecOf(opp);
      const venue = g.n ? "N" : (home ? "A" : "B");
      out.push({ opp, home, neutral: !!g.n, week: g.w, date: g.d, conf: g.c,
                 known, r: known ? predict(t, opp, venue) : null });
    }
    out.sort((a, b) => (a.week || 0) - (b.week || 0));
    return out;
  }

  /* League-wide distribution of each position group's win contribution, so a team's
     number can be read against the field instead of only against its own other
     groups. A bar chart of a team's own groups answers "where does this team get its
     wins", which is not the question - QB is worth more than TE everywhere, so QB's
     bar is longest for all 136 teams and the chart says nothing. Rank and distance
     from the median do say something. */
  const GROUP_LEAGUE = (function () {
    const acc = {};
    for (const t in players) {
      const bg = players[t].byGroup || {};
      for (const g in bg) (acc[g] = acc[g] || []).push({ t, v: bg[g] });
    }
    const out = {};
    for (const g in acc) {
      const arr = acc[g].slice().sort((a, b) => b.v - a.v);
      const vals = arr.map(x => x.v).sort((a, b) => a - b);
      const h = vals.length >> 1;
      const rank = {};
      arr.forEach((x, i) => { rank[x.t] = i + 1; });
      out[g] = {
        median: vals.length % 2 ? vals[h] : (vals[h - 1] + vals[h]) / 2,
        lo: vals[0], hi: vals[vals.length - 1], n: arr.length, rank,
      };
    }
    return out;
  })();

  const ord = n => n + (["th", "st", "nd", "rd"][(n % 100 - 20) % 10] ||
                        ["th", "st", "nd", "rd"][n % 100] || "th");

  /* One row per position group: value, a dot placed against the league median, the
     gap to that median, and the rank out of every rated team. */
  function groupRankHTML(t, byGroup, tint) {
    return GROUP_ORDER.filter(g => byGroup[g] != null).map(g => {
      const L = GROUP_LEAGUE[g], v = byGroup[g];
      if (!L) return "";
      const d = v - L.median;
      const span = Math.max(L.hi - L.median, L.median - L.lo) || 1;
      const x = Math.max(3, Math.min(97, 50 + 47 * d / span));
      const good = d >= 0, c = good ? tint : "var(--red)";
      const rk = L.rank[t];
      return `<div class="dp-row" title="${g}: ${v.toFixed(2)} wins, league median ${L.median.toFixed(2)}">
        <span class="dp-name ${OFF_GROUPS.has(g) ? "off" : "def"}">${g}</span>
        <span class="dp-val">${v.toFixed(2)}</span>
        <div class="dp-track"><i class="dp-med"></i>
          <i class="dp-span" style="left:${Math.min(50, x)}%;
             width:${Math.abs(x - 50)}%;background:${c}"></i>
          <i class="dp-dot" style="left:${x}%;background:${c}"></i></div>
        <span class="dp-delta ${good ? "pos" : "neg"}">${good ? "+" : "−"}${Math.abs(d).toFixed(2)}</span>
        <span class="dp-rank">${rk != null ? ord(rk) : "—"}<small>/${L.n}</small></span>
      </div>`;
    }).join("");
  }

  /* ---- toss-up games ----------------------------------------------------
     Which coin flips does this team actually have to win? The simulation already
     knows: every cell below is the subset of the 20,000 seasons where the games came
     out the way you set them, so the answer still obeys the bracket, the title games
     and the committee proxy. Unset games are averaged over, not assumed. */
  const tossState = {};

  function tossupHTML(t, tint, baseCFP) {
    const T = (cur().playoff.tossups || {})[t];
    if (!T || !T.games.length) return "";
    const k = T.games.length;
    const st = tossState[t] || (tossState[t] = new Array(k).fill(null));
    let n = 0, po = 0, wn = 0;
    for (let key = 0; key < (1 << k); key++) {
      let ok = true;
      for (let b = 0; b < k; b++) {
        const s = st[b];
        if (s != null && ((key >> b) & 1) !== s) { ok = false; break; }
      }
      if (!ok) continue;
      const c = T.n[key];
      if (!c) continue;
      n += c; po += T.playoff[key] * c; wn += T.wins[key] * c;
    }
    const cfp = n ? po / n : null, wins = n ? wn / n : null;
    const set = st.filter(s => s != null).length;
    const delta = cfp != null && baseCFP != null ? cfp - baseCFP : 0;

    const rows = T.games.map((g, b) => `
      <tr>
        <td class="num small">${g.w ?? ""}</td>
        <td class="loc">${g.home ? "vs" : "at"}</td>
        <td><div class="team-cell sm"><img src="${logoURL(g.opp)}" alt="" loading="lazy">
          <button class="team-link" data-team="${esc(g.opp)}">${esc(g.opp)}</button></div></td>
        <td class="num">${pct(g.p, 0)}</td>
        <td class="tu-cell">
          <div class="tu-seg" data-team="${esc(t)}" data-i="${b}">
            <button class="${st[b] === 1 ? "on w" : ""}" data-v="1">W</button>
            <button class="${st[b] == null ? "on" : ""}" data-v="">?</button>
            <button class="${st[b] === 0 ? "on l" : ""}" data-v="0">L</button>
          </div></td>
      </tr>`).join("");

    return `
      <div class="panel"><h3>Toss-up games
        <span class="hint">— the ${k} game${k > 1 ? "s" : ""} on this schedule the model
          calls closest to a coin flip</span></h3>
        <div class="tu-wrap">
          <div class="mini-wrap"><table class="mini"><thead><tr>
            <th class="num">Wk</th><th></th><th>Opponent</th>
            <th class="num">Win prob</th><th>Result</th>
          </tr></thead><tbody>${rows}</tbody></table></div>
          <div class="tu-out" style="--tint:${tint}">
            <div class="tu-big">${cfp == null ? "—" : pct(cfp, 0)}</div>
            <div class="tu-lab">chance to make the playoff</div>
            <div class="tu-delta ${delta >= 0 ? "pos" : "neg"}">
              ${set === 0 ? "season average" :
                `${delta >= 0 ? "+" : "−"}${Math.abs(100 * delta).toFixed(0)} pts vs average`}</div>
            <div class="tu-sub">${wins == null ? "" : wins.toFixed(1) + " projected wins"}</div>
            <div class="tu-sub dim">${n.toLocaleString()} matching simulations</div>
            <button class="tu-reset" data-team="${esc(t)}">Reset</button>
          </div>
        </div>
        <div class="wd-foot">Set any of these to a win or a loss and everything on the
          right is recomputed from the simulated seasons that actually broke that way
          &mdash; so it still respects conference title games, the committee ranking and
          the twelve-team format. Games left on <b>?</b> are averaged over.</div>
      </div>`;
  }

  function wireTossups() {
    document.querySelectorAll(".tu-seg button").forEach(b =>
      b.addEventListener("click", e => {
        const seg = e.currentTarget.parentElement;
        const t = seg.dataset.team, i = +seg.dataset.i, v = e.currentTarget.dataset.v;
        tossState[t][i] = v === "" ? null : +v;
        renderTeam();
      }));
    document.querySelectorAll(".tu-reset").forEach(b =>
      b.addEventListener("click", e => {
        const t = e.currentTarget.dataset.team;
        tossState[t] = tossState[t].map(() => null);
        renderTeam();
      }));
  }

  function renderTeam() {
    const t = selT.value;
    const R = ratingRow()[t] || {};
    const S = simRow()[t] || {};
    const tint = color(t);
    const dist = (cur().playoff.win_dist || {})[t];
    const roster = players[t];
    const sched = teamSchedule(t);

    /* ---- win distribution ---- */
    let distHTML = `<p class="sub">No simulated distribution for this team.</p>`;
    if (dist) {
      const total = dist.reduce((a, b) => a + b, 0) || 1;
      let lo = dist.findIndex(c => c / total > 0.004);
      let hi = dist.length - 1;
      if (lo < 0) lo = 0;
      while (hi > lo && dist[hi] / total <= 0.004) hi--;
      const slice = dist.slice(lo, hi + 1);
      const max = Math.max(...slice);
      const mode = lo + slice.indexOf(max);
      // Floor and ceiling: the central 80% of simulated seasons. A single projected
      // win total hides the thing people actually argue about, which is how far the
      // season can swing before anyone should be surprised.
      const q = f => {
        let c = 0;
        for (let w = 0; w < dist.length; w++) {
          c += dist[w];
          if (c / total >= f) return w;
        }
        return dist.length - 1;
      };
      const floorW = q(0.10), ceilW = q(0.90);
      distHTML = `<div class="wd">${slice.map((c, i) => {
        const w = lo + i, p = c / total;
        const inBand = w >= floorW && w <= ceilW;
        return `<div class="wd-col${inBand ? " in-band" : ""}"
          title="${w} wins — ${pct(p)} of simulations">
          <span class="wd-val">${p >= 0.03 ? pct(p, 0) : ""}</span>
          <div class="wd-bar" style="height:${Math.max(2, 100 * c / max)}%;
            background:${w === mode ? tint : (inBand ? rgba(t, .40) : rgba(t, .14))}"></div>
          <span class="wd-x">${w}</span></div>`;
      }).join("")}</div>
      <div class="wd-range" style="--tint:${tint}">
        <span class="wr-end">FLOOR<b>${floorW}</b></span>
        <span class="wr-mid">most likely <b style="color:${tint}">${mode}</b> wins
          &middot; mean ${(S.avg_wins ?? 0).toFixed(1)}</span>
        <span class="wr-end">CEILING<b>${ceilW}</b></span>
      </div>
      <div class="wd-foot">Regular-season wins across
        ${cur().playoff.n_sims.toLocaleString()} simulated seasons. Eight seasons in
        ten land between <b>${floorW}</b> and <b>${ceilW}</b> wins &mdash; the shaded
        columns. That spread comes from two things: games the model rates close, and
        the fact that the roster projection itself can be wrong${
          cur().playoff.talent_noise_sd
            ? `, which is simulated by moving every team's talent by
               ${cur().playoff.talent_noise_sd.toFixed(2)} of a standard deviation from
               season to season`
            : ""}.</div>`;
    }

    /* ---- player contributions ---- */
    let rosterHTML = `<p class="sub">No WAR projection available for this team.</p>`;
    if (roster) {
      const byGroup = roster.byGroup || {};
      const groupRows = groupRankHTML(t, byGroup, tint);

      const offTot = Object.entries(byGroup)
        .filter(([g]) => OFF_GROUPS.has(g)).reduce((s, [, v]) => s + v, 0);
      const defTot = (roster.winsTotal ?? roster.total) - offTot;
      rosterHTML = `
        <div class="od-split">
          <div class="od-chip2 off" style="--tint:${tint}">
            <b>${offTot.toFixed(2)}</b><span>wins from offense</span></div>
          <div class="od-chip2 def" style="--tint:${tint}">
            <b>${defTot.toFixed(2)}</b><span>wins from defense</span></div>
          <div class="od-chip2" style="--tint:${tint}">
            <b>${(roster.winsTotal ?? roster.total).toFixed(2)}</b><span>wins above replacement</span></div>
        </div>
        ${lineupHTML(t, roster, tint)}
        <h4 style="margin-top:22px">Every position group, against the rest of the country
          <span class="hint">— wins added, distance from the median FBS team, and rank</span></h4>
        <div class="dp">${groupRows}</div>
        <div class="wd-foot">The dot sits where this group falls relative to the
          <b>median FBS team</b> at the same position (the centre line). Quarterback
          is worth more than tight end at every school, so what matters is not how
          long a team's own bar is but how far it sits from everyone else's.</div>`;
    }

    /* ---- schedule ---- */
    const expWins = sched.reduce((s, g) => s + (g.r ? g.r.pA : FCS_WIN_P), 0);
    const schedHTML = sched.map(g => {
      const r = g.r;
      const win = r ? r.pA >= 0.5 : true;
      const loc = g.neutral ? "vs" : (g.home ? "vs" : "at");
      return `<tr class="${win ? "w" : "l"}">
        <td class="num small">${g.week ?? ""}</td>
        <td class="num small">${g.date ? g.date.slice(5) : ""}</td>
        <td class="loc">${loc}</td>
        <td><div class="team-cell sm">
          <img src="${logoURL(g.opp)}" alt="" loading="lazy">
          ${g.known ? `<button class="team-link" data-team="${esc(g.opp)}">${esc(g.opp)}</button>`
                    : `<span class="fcs">${esc(g.opp)}</span>`}
          ${g.conf ? `<span class="tag conf-tag">conf</span>` : ""}</div></td>
        <td class="num">${r ? displayScore(r).join("–") : "—"}</td>
        <td class="num">${r ? spreadLabel(r.margin) : "—"}</td>
        <td class="num"><div class="bar-wrap">
          <span class="pct">${r ? pct(r.pA, 0) : "95%"}</span>
          <div class="bar"><i style="width:${100 * (r ? r.pA : FCS_WIN_P)}%;background:${win ? tint : "var(--red)"}"></i></div>
        </div></td>
      </tr>`;
    }).join("");

    document.getElementById("team-body").innerHTML = `
      <div class="team-hero" style="--wash:${rgba(t, .16)};--tint:${tint}">
        <img class="hero-logo" src="${logoURL(t)}" alt="">
        <div class="hero-id">
          <div class="hero-name">${esc(t)}</div>
          <div class="hero-sub">${esc((meta[t] && meta[t].mascot) || "")} · ${esc(conf(t))}</div>
        </div>
        <div class="hero-stats">
          <div><b>#${R.rank ?? "—"}</b><span>power rank</span></div>
          <div><b>${S.avg_wins != null ? S.avg_wins.toFixed(1) + "–" + S.avg_losses.toFixed(1) : "—"}</b><span>proj record</span></div>
          <div><b>${pct(S.conf_champ ?? 0, 0)}</b><span>conf title</span></div>
          <div><b>${pct(S.playoff ?? 0, 0)}</b><span>make CFP</span></div>
          <div><b>${pct(S.champ ?? 0, 1)}</b><span>national title</span></div>
        </div>
      </div>

      <div class="panel"><h3>Projected win distribution</h3>${distHTML}</div>
      <div class="panel"><h3>Where the wins come from</h3>${rosterHTML}
        <div class="wd-foot">${roster && !roster.players ? "Position groups from" :
          "Projected starters from"} the 2026 two-deep, each carrying
          the wins ${roster && !roster.players ? "they add" : "he adds"} over a
          replacement-level player, against an average schedule. No rescaling is
          applied here: the build solves for the factor that puts summed team WAR in
          units of wins and applies it at source.${roster && roster.players
            ? ` <span class="tag unproven">?</span> marks a player with no prior FBS
              snaps, whose projection is a positional prior rather than a measurement.`
            : ""}</div></div>
      ${tossupHTML(t, tint, S.playoff)}
      <div class="panel"><h3>2026 schedule
        <span class="hint">— projected score, spread and win probability for every game</span></h3>
        <div class="mini-wrap"><table class="mini sched"><thead><tr>
          <th class="num">Wk</th><th class="num">Date</th><th></th><th>Opponent</th>
          <th class="num">Proj score</th><th class="num">Spread</th><th class="num">Win prob</th>
        </tr></thead><tbody>${schedHTML}</tbody></table></div>
        <div class="wd-foot">Win probabilities across the slate sum to
          <b style="color:${tint}">${expWins.toFixed(1)}</b> expected wins
          (non-FBS opponents counted at 95%). The Monte Carlo mean of
          <b>${(S.avg_wins ?? 0).toFixed(1)}</b> also includes a conference title game.</div>
      </div>`;
    wireTeamLinks();
    wireTossups();
  }
  selT.addEventListener("change", renderTeam);

  /* Any team name anywhere jumps to that team's breakdown. */
  function wireTeamLinks() {
    document.querySelectorAll(".team-link").forEach(a =>
      a.addEventListener("click", e => {
        const t = e.currentTarget.dataset.team;
        if (!vecOf(t)) return;
        selT.value = t;
        document.querySelectorAll(".tab").forEach(x =>
          x.classList.toggle("active", x.dataset.view === "team"));
        document.querySelectorAll(".view").forEach(v =>
          v.classList.toggle("active", v.id === "view-team"));
        renderTeam();
        window.scrollTo({ top: 0, behavior: "smooth" });
      }));
  }

  /* ---------- players ----------
     One flat table of every two-deep slot in FBS. The team page can only ever answer
     "is this player worth more than the model says", one roster at a time; the
     question people actually have is comparative - is this the best guard in the
     conference, who are the ten most valuable corners - and that needs every player in
     one sortable place. Editing here is the same store the team page writes to, so a
     change made in either shows up in both and in the ratings. */
  const ALL_PLAYERS = (function () {
    const rows = [];
    for (const [t, r] of Object.entries(players)) {
      if (!r || !r.players) continue;
      for (const p of r.players) rows.push({ t, conf: conf(t), ...p });
    }
    return rows;
  })();

  const plWar = r => { const e = WI.get(r.t, r.n); return e != null ? e : (r.raw || 0); };
  const plKey = r => r.t + "\u0000" + r.n;

  /* Depth, 2025 snaps and 2025 WAR are gone from this table. They are inputs to the
     projection rather than the projection, and four numeric columns meant the one
     column that is editable - and that everything else on the page moves with - was
     the last thing the eye reached. What replaces them answers the question the
     filters were being used to fake: where does this player sit, in FBS and at his
     own position.

     Both ranks are over EVERY projected slot, not the filtered view, and they are
     recomputed whenever an edit lands - a scenario that makes a backup the best
     quarterback in the country has to say so. */
  let plRankCache = null, plRankVer = -1;
  function plRanks() {
    if (plRankVer === WI.version() && plRankCache) return plRankCache;
    const overall = new Map(), pos = new Map(), groupN = {};
    ALL_PLAYERS.slice().sort((a, b) => plWar(b) - plWar(a)).forEach((r, i) => {
      overall.set(plKey(r), i + 1);
      const g = r.g || "—";
      groupN[g] = (groupN[g] || 0) + 1;
      pos.set(plKey(r), groupN[g]);
    });
    plRankCache = { overall, pos, groupN };
    plRankVer = WI.version();
    return plRankCache;
  }

  /* "RS JR" where the chart says redshirt junior, "JR" where it says junior. The two
     halves are stored separately (see export_viz) so the filters can ask about either
     one on its own; this puts them back together for display only. */
  const classLabel = r => r.c ? (r.rs ? `RS ${r.c}` : r.c) : "";

  const PL_COLS = [
    { k: "rk",   h: "#",        n: true, v: r => plRanks().overall.get(plKey(r)) },
    { k: "n",    h: "Player",   v: r => r.n },
    { k: "t",    h: "Team",     v: r => r.t },
    { k: "conf", h: "Conf",     v: r => r.conf },
    { k: "p",    h: "Pos",      v: r => r.p || r.g },
    { k: "c",    h: "Class",    v: r => classLabel(r) },
    { k: "prk",  h: "Pos rank", n: true, v: r => plRanks().pos.get(plKey(r)) },
    { k: "war",  h: "Proj WAR", n: true, v: r => plWar(r) },
  ];
  // Ranks sort smallest-first; every other numeric column sorts largest-first.
  const PL_ASC = new Set(["n", "t", "conf", "c", "rk", "prk"]);
  // The team page and this table used to disagree - the same player read 1.604 here
  // and 2.11 there - because the team page showed WAR after a display-time rescale
  // that this table did not apply. That rescale is gone (it was covering an attenuated
  // slope in build_hybrid, now fixed at source), so both surfaces show the same number
  // and there is nothing left to reconcile.
  let plSort = "war", plDesc = true;
  const PL_LIMIT = 300;

  function plFilled() {
    const q = document.getElementById("pl-search").value.trim().toLowerCase();
    const c = document.getElementById("pl-conf").value;
    const g = document.getElementById("pl-group").value;
    const cl = document.getElementById("pl-class").value;
    const tr = document.getElementById("pl-tr").value;
    let rows = ALL_PLAYERS;
    if (q) rows = rows.filter(r => r.n.toLowerCase().includes(q) ||
                                   r.t.toLowerCase().includes(q));
    if (c) rows = rows.filter(r => r.conf === c);
    if (g) rows = rows.filter(r => r.g === g);
    // A player with no class on the chart is excluded rather than silently kept - an
    // unknown class is not a match for JR.
    if (cl) rows = rows.filter(r => r.c === cl);
    // `tr` is "on a new team in 2026", not "has ever transferred" - see
    // build_roster_2026, which recomputes it as a change of team since 2025 so that
    // the flag the site shows is the one the model was trained on.
    if (tr === "tr") rows = rows.filter(r => r.tr === true);
    const col = PL_COLS.find(x => x.k === plSort) ||
                PL_COLS.find(x => x.k === "war");
    const val = col.v;
    return rows.slice().sort((a, b) => {
      const x = val(a), y = val(b);
      if (typeof x === "string") return plDesc ? y.localeCompare(x) : x.localeCompare(y);
      return plDesc ? y - x : x - y;
    });
  }

  function renderPlayers() {
    const all = plFilled();
    const rows = all.slice(0, PL_LIMIT);
    document.getElementById("pl-count").textContent =
      `${all.length > PL_LIMIT ? `top ${PL_LIMIT} of ` : ""}${all.length} slots` +
      (WI.count() ? ` · ${WI.count()} edited` : "");
    document.getElementById("pl-reset").disabled = !WI.count();

    const head = PL_COLS.map(c => `<th class="${c.n ? "num" : ""} sortable${
      c.k === plSort ? " sorted" : ""}" data-k="${c.k}">${c.h}${
      c.k === plSort ? (plDesc ? " ▾" : " ▴") : ""}</th>`).join("");

    const RK = plRanks();
    const body = rows.map(r => {
      const tint = color(r.t);
      const edited = WI.get(r.t, r.n) != null;
      const grp = r.g || "—";
      return `<tr class="${edited ? "wi-edited" : ""}">
        <td class="rank num">${RK.overall.get(plKey(r))}</td>
        <td><div class="team-cell sm"><span class="team-stripe" style="background:${tint}"></span>
          <span class="pl-name">${esc(r.n)}</span>${r.i
            ? ` <span class="tag unproven" title="No prior FBS snaps — this projection is a positional prior, not a measurement">?</span>` : ""}${r.out
            ? ` <span class="tag out" title="Out for the season — his snaps go to whoever replaces him, so his WAR is zero by construction">OUT</span>` : ""}</div></td>
        <td><div class="team-cell sm"><img src="${logoURL(r.t)}" alt="" loading="lazy">
          <button class="team-link" data-team="${esc(r.t)}">${esc(r.t)}</button></div></td>
        <td class="pl-conf">${esc(r.conf)}</td>
        <td><span class="wi-pos">${esc(r.p || r.g)}</span></td>
        <td class="pl-conf">${esc(classLabel(r) || "—")}${r.tr
          ? ` <span class="tag tr" title="Transferred in for 2026">TR</span>` : ""}</td>
        <td class="num pl-prk">${RK.pos.get(plKey(r))}<span class="pl-of">of ${
          (RK.groupN[grp] || 0).toLocaleString()} ${esc(grp)}</span></td>
        <td class="num"><input class="wi-in pl-in" type="number" step="0.05"
          data-team="${esc(r.t)}" data-player="${esc(r.n)}"
          value="${plWar(r).toFixed(3)}" aria-label="Projected WAR for ${esc(r.n)}"></td>
      </tr>`;
    }).join("");

    document.getElementById("pl-table").innerHTML =
      `<div class="mini-wrap pl-scroll"><table class="mini pl-table"><thead><tr>${head}</tr></thead>
       <tbody>${body}</tbody></table></div>` +
      (all.length > PL_LIMIT ? `<div class="wd-foot">Showing the top ${PL_LIMIT} by the
        current sort. Narrow with the filters or the search box to see the rest.</div>` : "");
    wirePlayers();
  }

  function wirePlayers() {
    document.querySelectorAll("#pl-table th.sortable").forEach(th =>
      th.addEventListener("click", () => {
        const k = th.dataset.k;
        if (k === plSort) plDesc = !plDesc;
        else { plSort = k; plDesc = !PL_ASC.has(k); }
        renderPlayers();
      }));
    // The whole table is replaced on every edit, so the caret would jump to the top of
    // the page and the row being worked on would scroll away. Keep both.
    document.querySelectorAll("#pl-table .pl-in").forEach(el =>
      el.addEventListener("change", () => {
        // The table scrolls inside .pl-scroll, not with the page, so it is that
        // container's scrollTop that has to survive the re-render - restoring
        // window.scrollY alone left the row you just edited somewhere off screen.
        const box = document.querySelector(".pl-scroll");
        const y = window.scrollY, inner = box ? box.scrollTop : 0;
        const key = el.dataset.team + "|" + el.dataset.player;
        const v = parseFloat(el.value);
        WI.set(el.dataset.team, el.dataset.player, isFinite(v) ? v : null);
        renderAll();
        window.scrollTo(0, y);
        const box2 = document.querySelector(".pl-scroll");
        if (box2) box2.scrollTop = inner;
        const again = [...document.querySelectorAll("#pl-table .pl-in")]
          .find(e => e.dataset.team + "|" + e.dataset.player === key);
        // preventScroll: focus() scrolls its target into view by default, which undid
        // the scroll restore above and threw the page thousands of pixels down the
        // table - the row had usually moved on the re-sort, so it scrolled to wherever
        // the edit sent it.
        if (again) again.focus({ preventScroll: true });
      }));
    // wireTeamLinks() binds .team-link globally but only runs from renderTeam, by
    // which point this table may not exist yet - so these are bound here, to the same
    // effect: switch to the team tab and show that roster.
    document.querySelectorAll("#pl-table .team-link").forEach(b =>
      b.addEventListener("click", () => {
        const t = b.dataset.team;
        if (!vecOf(t)) return;
        selT.value = t;
        document.querySelectorAll(".tab").forEach(x =>
          x.classList.toggle("active", x.dataset.view === "team"));
        document.querySelectorAll(".view").forEach(v =>
          v.classList.toggle("active", v.id === "view-team"));
        renderTeam();
        window.scrollTo({ top: 0, behavior: "smooth" });
      }));
  }

  function fillPlayerSelects() {
    const cs = document.getElementById("pl-conf"), gs = document.getElementById("pl-group");
    if (cs.options.length <= 1) {
      [...new Set(ALL_PLAYERS.map(r => r.conf))].filter(Boolean).sort()
        .forEach(c => cs.add(new Option(c, c)));
    }
    if (gs.options.length <= 1) {
      // GROUP_ORDER, not sorted: alphabetical put CB and DT above QB, which is not an
      // order anyone reads a roster in. This is offence then defence, each front to
      // back.
      const have = new Set(ALL_PLAYERS.map(r => r.g));
      GROUP_ORDER.filter(g => have.has(g)).forEach(g => gs.add(new Option(g, g)));
    }
  }

  ["pl-search", "pl-conf", "pl-group", "pl-class", "pl-tr"].forEach(id =>
    document.getElementById(id).addEventListener("input", renderPlayers));
  document.getElementById("pl-reset").addEventListener("click", () => {
    WI.reset(); renderAll();
  });

  /* =======================================================================
     TEAM RATINGS BY INPUT
     =======================================================================
     The Season Odds table ranks teams by the one number the model produces. This
     ranks them by the numbers it is BUILT from, which is a different and often more
     useful question: who has the best quarterback room, the best offensive line, the
     toughest schedule. The team page answers all of that for one team at a time and
     could never answer "who is best", because that comparison does not exist on a
     page about a single team.

     Two column sets, because they are two different kinds of number and averaging
     them into one table would invite adding a z-score to a win. Roster WAR is in
     wins; the model inputs are the standardised features the coefficients multiply. */
  const TR_OFF = GROUP_ORDER.filter(g => OFF_GROUPS.has(g));
  const TR_DEF = GROUP_ORDER.filter(g => !OFF_GROUPS.has(g));

  /* Per-team WAR by position group, summed live from the same rows the Players tab
     edits - NOT from players.json's `byGroup`, which is the unedited baseline and
     would have this table disagreeing with every other surface the moment anyone
     changed a projection. */
  let trWarCache = null, trWarVer = -1;
  function trGroupWar() {
    if (trWarVer === WI.version() && trWarCache) return trWarCache;
    const out = {};
    for (const r of ALL_PLAYERS) {
      const o = out[r.t] || (out[r.t] = {});
      const g = r.g || "—";
      o[g] = (o[g] || 0) + plWar(r);
    }
    // GROUP_ORDER partitions the two-deep exactly, so off + def is the whole roster
    // and matches the total the team page shows.
    for (const t in out) {
      const o = out[t];
      o.__off = TR_OFF.reduce((s, g) => s + (o[g] || 0), 0);
      o.__def = TR_DEF.reduce((s, g) => s + (o[g] || 0), 0);
      o.__all = o.__off + o.__def;
    }
    trWarCache = out; trWarVer = WI.version();
    return trWarCache;
  }

  const TR_SETS = {
    war: {
      label: "Roster WAR by position",
      note: `Projected wins above replacement contributed by each position group,
        summed over that group's slots in the 2026 two-deep. These are the same
        per-player figures the <b>Players</b> tab lists and edits, so an edit there
        re-ranks this table.`,
      fmt: v => v.toFixed(2),
      cols: () => [
        ...GROUP_ORDER.map(g => ({ k: g, h: g,
          cls: OFF_GROUPS.has(g) ? "tr-off" : "tr-def" })),
        { k: "__off", h: "Offense", cls: "tr-off tr-sum" },
        { k: "__def", h: "Defense", cls: "tr-def tr-sum" },
        { k: "__all", h: "Total",   cls: "tr-sum" },
      ],
      val: (t, k) => (trGroupWar()[t] || {})[k] ?? 0,
      has: t => !!trGroupWar()[t],
    },
    inputs: {
      label: "Model inputs",
      note: `The standardised features the trained coefficients actually multiply.
        <b>Power</b> is the output &mdash; mean neutral-site win probability against
        the rest of FBS &mdash; and the five underneath are what produce it. Schedule
        strength is the mean rating of everyone on the slate, so a high number is a
        hard season, not a good team.`,
      fmt: (v, k) => k === "power" ? (100 * v).toFixed(1) : v.toFixed(2),
      cols: () => [
        { k: "power",     h: "Power",     cls: "tr-sum" },
        { k: "O",         h: "Offense",   cls: "tr-off" },
        { k: "D",         h: "Defense",   cls: "tr-def" },
        { k: "talent",    h: "Talent" },
        { k: "returning", h: "Returning" },
        { k: "sos",       h: "Schedule" },
      ],
      val: (t, k) => { const r = trRow()[t]; return r && r[k] != null ? r[k] : 0; },
      has: t => !!trRow()[t],
    },
  };
  const trRow = () => ratingRow();
  let trSet = "war", trSort = "__all", trDesc = true;

  function trRows() {
    const S = TR_SETS[trSet];
    const q = document.getElementById("tr-search").value.trim().toLowerCase();
    const c = document.getElementById("tr-conf").value;
    const tier = document.getElementById("tr-tier").value;
    let rows = liveRatings().filter(r => S.has(r.team));
    if (q) rows = rows.filter(r => r.team.toLowerCase().includes(q) ||
      ((meta[r.team] && meta[r.team].mascot) || "").toLowerCase().includes(q));
    if (c) rows = rows.filter(r => r.conference === c);
    if (tier === "p4") rows = rows.filter(r => P4.has(r.conference));
    if (tier === "g6") rows = rows.filter(r => G6.has(r.conference));
    return rows;
  }

  function renderRatings() {
    const host = document.getElementById("tr-table");
    if (!host) return;
    const S = TR_SETS[trSet];
    const cols = S.cols();
    if (!cols.some(c => c.k === trSort)) trSort = cols[cols.length - 1].k;

    // Ranked over every team in the set, not the filtered view: filtering to one
    // conference should tell you where its teams sit in FBS, not renumber them 1-16.
    const all = liveRatings().filter(r => S.has(r.team));
    const rank = {};
    all.slice().sort((a, b) => S.val(b.team, trSort) - S.val(a.team, trSort))
       .forEach((r, i) => { rank[r.team] = i + 1; });

    // The sorted column is the only one that gets a fill. Thirteen bars in one row
    // would be a heat map nobody asked for; one bar says "this is the column you are
    // ranking by" and stays legible.
    const vals = all.map(r => S.val(r.team, trSort));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const span = (hi - lo) || 1;

    const rows = trRows().slice()
      .sort((a, b) => trDesc ? S.val(b.team, trSort) - S.val(a.team, trSort)
                             : S.val(a.team, trSort) - S.val(b.team, trSort));
    document.getElementById("tr-count").textContent =
      `${rows.length} of ${all.length} teams`;

    const head = `<th class="num">#</th><th>Team</th>` + cols.map(c =>
      `<th class="num sortable ${c.cls || ""}${c.k === trSort ? " sorted" : ""}"
        data-k="${c.k}">${c.h}${c.k === trSort ? (trDesc ? " ▾" : " ▴") : ""}</th>`).join("");

    const body = rows.map(r => {
      const t = r.team, tint = color(t);
      const cells = cols.map(c => {
        const v = S.val(t, c.k);
        const fig = S.fmt(v, c.k);
        if (c.k !== trSort) {
          return `<td class="num ${c.cls || ""}">${fig}</td>`;
        }
        return `<td class="num ${c.cls || ""} sorted"><div class="bar-wrap">
          <span class="pct">${fig}</span>
          <div class="bar"><i style="width:${100 * (v - lo) / span}%;
            background:${tint}"></i></div></div></td>`;
      }).join("");
      return `<tr><td class="rank num">${rank[t]}</td>
        <td><div class="team-cell sm">
          <span class="team-stripe" style="background:${tint}"></span>
          <img src="${logoURL(t)}" alt="" loading="lazy">
          <div><button class="team-link" data-team="${esc(t)}">${esc(t)}</button>
            <div class="conf">${esc(r.conference)}</div></div></div></td>${cells}</tr>`;
    }).join("");

    host.innerHTML =
      `<div class="mini-wrap tr-scroll"><table class="mini tr-table">
         <thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>
       <div class="wd-foot">${S.note}</div>`;

    host.querySelectorAll("th.sortable").forEach(th =>
      th.addEventListener("click", () => {
        const k = th.dataset.k;
        if (k === trSort) trDesc = !trDesc;
        else { trSort = k; trDesc = true; }
        renderRatings();
      }));
    wireTeamLinks();
  }

  function fillRatingSelects() {
    const sel = document.getElementById("tr-conf");
    if (!sel || sel.options.length > 1) return;
    [...new Set(cur().ratings.teams.map(t => t.conference))].sort()
      .forEach(c => sel.add(new Option(c, c)));
  }

  ["tr-search", "tr-conf", "tr-tier"].forEach(id =>
    document.getElementById(id).addEventListener("input", renderRatings));
  document.getElementById("tr-set").addEventListener("change", e => {
    trSet = e.target.value;
    // Each set has its own natural default: total roster wins, or the power rating
    // the other five inputs feed. Carrying a sort key across sets is impossible
    // anyway - they share no column names.
    trSort = trSet === "war" ? "__all" : "power";
    trDesc = true;
    renderRatings();
  });


  /* ---------- boot ---------- */
  function render(view) {
    if (view === "dash") renderDash();
    else if (view === "playoff") renderPlayoff();
    else if (view === "matchup") renderMatchup();
    else if (view === "team") renderTeam();
    else if (view === "scenario") renderScenario();
    else if (view === "ratings") renderRatings();
    else if (view === "players") renderPlayers();
  }
  function renderAll() {
    fillConfSelect(); fillPlayerSelects(); fillRatingSelects();
    fillScenarioSelects();
    renderDash(); renderPlayoff(); renderScenario(); renderMatchup(); renderTeam();
    renderRatings(); renderPlayers();
  }
  // Check the client-side power reproduces the exported column before anyone edits
  // anything. A scenario is only meaningful as a difference from the published
  // baseline, so if the baseline does not agree the whole panel is theatre.
  if (WI.enabled()) verifyPower();
  renderAll();
})();
