/* CFB Model v2 — 2026 visuals.
   Tabs: Top 25 · Playoff Projection · Matchup Simulator · Ratings Lab (editor).
   Two rating lenses: "" (balanced) and "_roster" (roster-weighted).

   The trained model math is frozen (coefficients in model*.json). The Ratings Lab
   lets you edit each team's 6 model INPUTS; power ratings, the matchup sim, and a
   re-runnable playoff Monte Carlo all recompute from those edits, client-side. */
(async function () {
  const fetchJSON = u => fetch(u).then(r => r.json());
  const [teams, schedule, ...sets] = await Promise.all([
    fetchJSON("data/teams.json"),
    fetchJSON("data/schedule.json"),
    ...["", "_roster"].flatMap(s =>
      ["ratings", "playoff", "model"].map(f => fetchJSON(`data/${f}${s}.json`))),
  ]);
  const DATA = {
    "": { ratings: sets[0], playoff: sets[1], model: sets[2] },
    "_roster": { ratings: sets[3], playoff: sets[4], model: sets[5] },
  };
  let lens = "";
  const cur = () => DATA[lens];
  // Live sim results per lens (start with the backend 20k JSON; replaced on re-sim).
  const simData = { "": DATA[""].playoff, "_roster": DATA["_roster"].playoff };

  /* ---------- team metadata helpers ---------- */
  const meta = teams;
  const logoURL = t => meta[t] ? "logos/" + encodeURIComponent(meta[t].logo) : "";
  const conf = t => (meta[t] && meta[t].conference) || "—";
  function color(t) {
    let c = (meta[t] && meta[t].color) || "#58a6ff";
    let [r, g, b] = [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
    if (isNaN(r)) return "#58a6ff";
    while (0.299 * r + 0.587 * g + 0.114 * b < 90) {
      r = Math.min(255, r + 40); g = Math.min(255, g + 40); b = Math.min(255, b + 40);
    }
    return `rgb(${r},${g},${b})`;
  }
  const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

  /* ---------- editable inputs / overrides ---------- */
  // FIELD index -> label. Vector order: [O, D, fp_margin, pythag, talent, returning]
  const FIELDS = [
    { i: 0, label: "Off", grp: "perf" }, { i: 1, label: "Def", grp: "perf" },
    { i: 2, label: "FieldPos", grp: "perf" }, { i: 3, label: "Pythag", grp: "perf" },
    { i: 4, label: "Talent", grp: "roster" }, { i: 5, label: "Returning", grp: "roster" },
  ];
  const LS_KEY = "cfb_overrides_v1";
  let overrides = {};
  try { overrides = JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch (e) {}
  overrides[""] = overrides[""] || {};
  overrides["_roster"] = overrides["_roster"] || {};
  const saveOverrides = () => {
    try { localStorage.setItem(LS_KEY, JSON.stringify(overrides)); } catch (e) {}
  };

  // Roster editing (only when the Python server is present). rosterEdits holds the
  // user's edited depth charts per lens; the server turns them into frame vectors.
  const RE_KEY = "cfb_roster_edits_v1";
  let ROSTERS = null, apiAvailable = false;
  let rosterEdits = {};
  try { rosterEdits = JSON.parse(localStorage.getItem(RE_KEY)) || {}; } catch (e) {}
  rosterEdits[""] = rosterEdits[""] || {};
  rosterEdits["_roster"] = rosterEdits["_roster"] || {};
  const saveRosterEdits = () => {
    try { localStorage.setItem(RE_KEY, JSON.stringify(rosterEdits)); } catch (e) {}
  };
  const serverLens = () => (lens === "_roster" ? "roster" : "");
  async function postRecompute() {
    const edits = rosterEdits[lens];
    const res = await fetch("/api/recompute", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lens: serverLens(), edits }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    rosterVecs[lens] = Object.keys(edits).length ? data.teams : null;
  }

  const names = () => Object.keys(cur().model.teams);      // all sim teams (138)
  // Rankable teams = those the backend rated (have real 2025 priors, 136). The 2
  // brand-new FBS teams exist only for the schedule/sim, not the power ranking.
  const rankNames = () => cur().ratings.teams.map(t => t.team);

  // Roster layer: when the Python server re-derives inputs from edited depth
  // charts, it returns a full frame that replaces the baseline vectors.
  const rosterVecs = { "": null, "_roster": null };
  const baseVec = t => (rosterVecs[lens] && rosterVecs[lens][t]) || cur().model.teams[t];
  function effVec(t) {                    // baseline (+ roster layer) + direct field edits
    const v = baseVec(t).slice();
    const o = overrides[lens][t];
    if (o) for (const k in o) v[+k] = o[k];
    return v;
  }
  const isEdited = (t, i) => overrides[lens][t] && overrides[lens][t][i] !== undefined;
  const teamEditCount = t => overrides[lens][t] ? Object.keys(overrides[lens][t]).length : 0;
  const totalEdits = () => Object.values(overrides[lens]).reduce((s, o) => s + Object.keys(o).length, 0);

  /* ---------- frozen win-prob math (verified equal to Python) ---------- */
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

  // win prob for A over B given the 6-diff x and home flag for A (0/1)
  function winpFromDiff(x, homeA) {
    const M = cur().model, L = M.logistic, G = M.margin;
    const z = L.intercept + dot(L.coef, x) + homeA * L.hfa;
    const m = G.intercept + dot(G.coef, x) + homeA * G.hfa;
    return M.ens_w * sigmoid(z) + (1 - M.ens_w) * normCdf(m / G.sigma);
  }
  const diffVec = (a, b) => [a[0]-b[1], a[1]-b[0], a[2]-b[2], a[3]-b[3], a[4]-b[4], a[5]-b[5]];

  /* ---------- live power ratings (neutral round-robin) ---------- */
  let power = {}, rankOf = {};
  function computePower() {
    const ns = rankNames(), n = ns.length;
    const vecs = ns.map(effVec);
    const sum = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const p = winpFromDiff(diffVec(vecs[i], vecs[j]), 0); // neutral
        sum[i] += p; sum[j] += (1 - p);
      }
    }
    power = {}; const arr = [];
    for (let i = 0; i < n; i++) { power[ns[i]] = sum[i] / (n - 1); arr.push(ns[i]); }
    arr.sort((a, b) => power[b] - power[a]);
    rankOf = {}; arr.forEach((t, k) => rankOf[t] = k + 1);
    return arr;
  }

  /* ---------- tabs + lens ---------- */
  document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === b));
    document.querySelectorAll(".view").forEach(v =>
      v.classList.toggle("active", v.id === "view-" + b.dataset.view));
    // Re-render the view being shown so it reflects any Ratings Lab edits.
    const view = b.dataset.view;
    if (view === "top25") renderTop25();
    else if (view === "playoff") renderPlayoff();
    else if (view === "matchup") renderMatchup();
  }));
  document.querySelectorAll(".lens-opt").forEach(b => b.addEventListener("click", () => {
    lens = b.dataset.lens;
    document.querySelectorAll(".lens-opt").forEach(x => x.classList.toggle("active", x === b));
    computePower();
    renderAll();
    renderLab();
  }));

  /* ---------- Top 25 ---------- */
  function renderTop25() {
    const all = document.getElementById("showAll").checked;
    const order = Object.keys(power).sort((a, b) => power[b] - power[a]);
    const list = all ? order : order.slice(0, 25);
    const maxPower = power[order[0]];
    const otherLens = lens === "" ? "_roster" : "";
    const sim = Object.fromEntries((simData[lens].teams || []).map(t => [t.team, t]));
    // rank under the other lens (baseline, no live edits) for the move arrows
    const otherRank = Object.fromEntries(
      (DATA[otherLens].ratings.teams).map(t => [t.team, t.rank]));
    const rows = list.map(t => {
      const v = effVec(t), pw = power[t], rk = rankOf[t];
      const d = (otherRank[t] || rk) - rk;
      const move = d === 0 ? "" :
        `<span class="move ${d > 0 ? "up" : "down"}">${d > 0 ? "▲" : "▼"}${Math.abs(d)}</span>`;
      const s = sim[t] || {};
      return `
      <tr class="${rk <= 4 ? "top4" : ""}">
        <td class="rank">${rk}</td>
        <td><div class="team-cell"><img src="${logoURL(t)}" alt="">
          <div><div>${esc(t)} ${move}</div><div class="conf">${esc(conf(t))}</div></div>
        </div></td>
        <td><div class="bar-wrap"><div class="bar">
            <i class="${rk <= 4 ? "gold" : ""}" style="width:${100 * pw / maxPower}%"></i>
          </div><span class="pct">${(100 * pw).toFixed(1)}</span></div></td>
        <td class="num"><span class="od-chip ${v[0] >= 0 ? "pos" : "neg"}">O ${v[0] >= 0 ? "+" : ""}${v[0].toFixed(2)}</span></td>
        <td class="num"><span class="od-chip ${v[1] >= 0 ? "pos" : "neg"}">D ${v[1] >= 0 ? "+" : ""}${v[1].toFixed(2)}</span></td>
        <td class="num">${s.avg_wins != null ? s.avg_wins.toFixed(1) + "–" + s.avg_losses.toFixed(1) : "—"}</td>
        <td class="num">${s.playoff != null ? pct(s.playoff) : "—"}</td>
      </tr>`;
    }).join("");
    document.getElementById("top25-table").innerHTML = `
      <table><thead><tr>
        <th></th><th>Team</th><th>Power</th><th class="num">Offense</th>
        <th class="num">Defense</th><th class="num">Proj Record</th><th class="num">CFP&nbsp;%</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  }
  document.getElementById("showAll").addEventListener("change", renderTop25);

  /* ---------- Playoff render (from a sim-result object) ---------- */
  function renderPlayoff() {
    const playoff = simData[lens];
    const custom = playoff.custom;
    const lensNote = lens === "_roster"
      ? "Lens: <b>roster-weighted</b> — 70% two-deep PFF talent, full continuity shrinkage."
      : "Lens: <b>balanced</b> — the best-backtesting blend of 2025 results and 2026 roster signals.";
    const editNote = totalEdits()
      ? ` <b style="color:var(--accent)">${totalEdits()} custom input edit(s) active</b> — click “Re-simulate” to fold them into these odds.`
      : "";
    document.getElementById("playoff-meta").innerHTML =
      `Monte Carlo over the full 2026 schedule — <b>${playoff.n_sims.toLocaleString()}
       simulations</b>${custom ? " (your inputs)" : ""}. Rules: ${esc(playoff.rules)}.
       Committee proxy calibrated on the final 2025 CFP ranking (&rho; = 0.92). ${lensNote}${editNote}`;

    const P = playoff.teams, field = P.slice(0, 12), s = i => field[i];
    const slot = (t, seedNo, prob) => `
      <div class="slot"><span class="seed">${seedNo}</span>
        <img src="${logoURL(t.team)}" alt=""> ${esc(meta[t.team]?.abbreviation || t.team)}
        <span class="p">${prob != null ? pct(prob, 0) : ""}</span></div>`;
    document.getElementById("bracket").innerHTML = `
      <div class="round-label">First Round</div><div class="round-label">Quarterfinals</div>
      <div class="round-label">Semifinals</div><div class="round-label">Championship</div>
      <div class="round-col">
        <div class="game-card">${slot(s(7), 8, s(7).playoff)}${slot(s(8), 9, s(8).playoff)}</div>
        <div class="game-card">${slot(s(4), 5, s(4).playoff)}${slot(s(11), 12, s(11).playoff)}</div>
        <div class="game-card">${slot(s(5), 6, s(5).playoff)}${slot(s(10), 11, s(10).playoff)}</div>
        <div class="game-card">${slot(s(6), 7, s(6).playoff)}${slot(s(9), 10, s(9).playoff)}</div>
      </div>
      <div class="round-col">
        <div class="game-card bye-card">${slot(s(0), 1, s(0).bye)}<span class="bye-tag">BYE · vs 8/9 winner</span></div>
        <div class="game-card bye-card">${slot(s(3), 4, s(3).bye)}<span class="bye-tag">BYE · vs 5/12 winner</span></div>
        <div class="game-card bye-card">${slot(s(2), 3, s(2).bye)}<span class="bye-tag">BYE · vs 6/11 winner</span></div>
        <div class="game-card bye-card">${slot(s(1), 2, s(1).bye)}<span class="bye-tag">BYE · vs 7/10 winner</span></div>
      </div>
      <div class="round-col">
        <div class="game-card"><span class="bye-tag">SEMIFINAL 1</span>
          ${slot(s(0), 1, s(0).sf)}${slot(s(3), 4, s(3).sf)}</div>
        <div class="game-card"><span class="bye-tag">SEMIFINAL 2</span>
          ${slot(s(1), 2, s(1).sf)}${slot(s(2), 3, s(2).sf)}</div>
      </div>
      <div class="round-col">
        <div class="game-card trophy-card"><div class="emoji">🏆</div>
          ${slot(s(0), 1, s(0).champ)}${slot(s(1), 2, s(1).champ)}
          <span class="bye-tag">title odds shown</span></div>
      </div>`;

    const maxCFP = Math.max(...P.map(t => t.playoff));
    document.getElementById("playoff-table").innerHTML = `
      <table><thead><tr>
        <th></th><th>Team</th><th>Proj Record</th><th>Conf&nbsp;Champ</th>
        <th>Make&nbsp;CFP</th><th class="num">Bye</th><th class="num">Semis</th>
        <th class="num">Final</th><th class="num">Natty</th>
      </tr></thead><tbody>${P.slice(0, 40).map((t, i) => `
        <tr>
          <td class="rank">${i + 1}</td>
          <td><div class="team-cell"><img src="${logoURL(t.team)}" alt="">
            <div><div>${esc(t.team)}</div><div class="conf">${esc(t.conference)}</div></div>
          </div></td>
          <td class="num">${t.avg_wins.toFixed(1)}–${t.avg_losses.toFixed(1)}</td>
          <td><div class="bar-wrap"><div class="bar"><i class="green" style="width:${100 * t.conf_champ}%"></i></div>
            <span class="pct">${pct(t.conf_champ)}</span></div></td>
          <td><div class="bar-wrap"><div class="bar"><i style="width:${100 * t.playoff / maxCFP}%"></i></div>
            <span class="pct">${pct(t.playoff)}</span></div></td>
          <td class="num">${pct(t.bye)}</td>
          <td class="num">${pct(t.sf)}</td>
          <td class="num">${pct(t.final)}</td>
          <td class="num"><b>${pct(t.champ)}</b></td>
        </tr>`).join("")}</tbody></table>`;
  }

  /* ---------- Playoff Monte Carlo (JS port of scripts/simulate_playoff.py) ---------- */
  const P4 = new Set(["ACC", "Big 12", "Big Ten", "SEC"]);
  const G6 = new Set(["American Athletic", "Conference USA", "Mid-American",
    "Mountain West", "Pac-12", "Sun Belt"]);
  const CCG = new Set([...P4, ...G6]);
  const W_WINPCT = 10, W_RATING = 1.0, W_SOS = 0.75, FCS_WIN_P = 0.95, FCS_OPP = -2.0;

  function buildSimContext() {
    const ns = names(), n = ns.length, idx = {};
    ns.forEach((t, i) => idx[t] = i);
    const vecs = ns.map(effVec);
    // rating z = z-score of (O + D)/2
    let rating = vecs.map(v => (v[0] + v[1]) / 2);
    const rm = rating.reduce((a, b) => a + b, 0) / n;
    const rsd = Math.sqrt(rating.reduce((a, b) => a + (b - rm) ** 2, 0) / n) || 1;
    rating = rating.map(x => (x - rm) / rsd);
    // pairwise neutral + home win-prob matrices (home-team perspective)
    const Pn = new Float32Array(n * n), Ph = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const x = diffVec(vecs[i], vecs[j]);
      Pn[i * n + j] = winpFromDiff(x, 0);
      Ph[i * n + j] = winpFromDiff(x, 1);
    }
    // schedule split
    const gH = [], gA = [], gP = [], gConf = [];
    const buyTeam = [];
    const sosSum = new Float64Array(n), sosN = new Float64Array(n);
    const fcsCount = new Float64Array(n);
    for (const g of schedule) {
      const hi = idx[g.h], ai = idx[g.a];
      if (hi !== undefined && ai !== undefined) {
        gH.push(hi); gA.push(ai);
        gP.push(g.n ? Pn[hi * n + ai] : Ph[hi * n + ai]);
        gConf.push(conf(g.h) === conf(g.a) ? 1 : 0);
        sosSum[hi] += rating[ai]; sosN[hi]++;
        sosSum[ai] += rating[hi]; sosN[ai]++;
      } else if (hi !== undefined || ai !== undefined) {
        const t = hi !== undefined ? hi : ai;
        buyTeam.push(t); fcsCount[t]++;
        sosSum[t] += FCS_OPP; sosN[t]++;
      }
    }
    const sos = Array.from({ length: n }, (_, i) => sosSum[i] / Math.max(1, sosN[i]));
    const sm = sos.reduce((a, b) => a + b, 0) / n;
    const ssd = Math.sqrt(sos.reduce((a, b) => a + (b - sm) ** 2, 0) / n) || 1;
    const sosZ = sos.map(x => (x - sm) / ssd);
    const staticScore = Array.from({ length: n }, (_, i) => W_RATING * rating[i] + W_SOS * sosZ[i]);
    // conference games count + members
    const confGames = new Float64Array(n), baseGames = new Float64Array(n);
    for (let k = 0; k < gH.length; k++) {
      baseGames[gH[k]]++; baseGames[gA[k]]++;
      if (gConf[k]) { confGames[gH[k]]++; confGames[gA[k]]++; }
    }
    for (const t of buyTeam) baseGames[t]++;
    const confMembers = {};
    ns.forEach((t, i) => { const c = conf(t); if (CCG.has(c)) (confMembers[c] = confMembers[c] || []).push(i); });
    return { ns, n, idx, rating, Pn, gH, gA, gP, gConf, buyTeam, fcsCount,
      staticScore, confGames, baseGames, confMembers };
  }

  function runSimChunk(ctx, nSims, acc) {
    const { n, gH, gA, gP, gConf, buyTeam, fcsCount, staticScore,
      confGames, baseGames, confMembers, Pn, rating } = ctx;
    const confList = Object.entries(confMembers);
    const g6pool = [];
    ctx.ns.forEach((t, i) => { if (G6.has(conf(t))) g6pool.push(i); });
    for (let s = 0; s < nSims; s++) {
      const w = new Float64Array(n), cw = new Float64Array(n);
      for (let k = 0; k < gH.length; k++) {
        if (Math.random() < gP[k]) { w[gH[k]]++; if (gConf[k]) cw[gH[k]]++; }
        else { w[gA[k]]++; if (gConf[k]) cw[gA[k]]++; }
      }
      for (const t of buyTeam) if (Math.random() < FCS_WIN_P) w[t]++;
      const gp = Float64Array.from(baseGames);
      const champs = {};
      for (const [c, mem] of confList) {
        let b1 = -1, b2 = -1, s1 = -Infinity, s2 = -Infinity;
        for (const t of mem) {
          const sc = cw[t] / Math.max(1, confGames[t]) + 1e-4 * rating[t];
          if (sc > s1) { s2 = s1; b2 = b1; s1 = sc; b1 = t; }
          else if (sc > s2) { s2 = sc; b2 = t; }
        }
        const p = Pn[b1 * n + b2];
        const win = Math.random() < p ? b1 : b2;
        champs[c] = win; w[win]++; gp[b1]++; gp[b2]++;
        acc.confChamp[win]++;
      }
      const score = new Float64Array(n);
      for (let t = 0; t < n; t++) score[t] = W_WINPCT * (w[t] / gp[t]) + staticScore[t];
      // selection: 4 P4 champs + best-scoring G6 team, then fill by score
      const field = new Set();
      for (const c of P4) field.add(champs[c]);
      let bestG6 = -1, bg = -Infinity;
      for (const t of g6pool) if (score[t] > bg) { bg = score[t]; bestG6 = t; }
      field.add(bestG6);
      const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => score[b] - score[a]);
      for (const t of order) { if (field.size >= 12) break; field.add(t); }
      const seeds = order.filter(t => field.has(t)); // score-sorted, length 12
      for (let r = 0; r < 12; r++) { acc.playoff[seeds[r]]++; if (r < 4) acc.bye[seeds[r]]++; }
      // bracket (straight seeding, fixed) — mirrors the Python
      const playN = (a, b) => Math.random() < Pn[a * n + b] ? a : b;
      // first round hosted by higher seed (home matrix)
      const fr = (hi, lo) => Math.random() < ctx.homeP(hi, lo) ? hi : lo;
      const a89 = fr(seeds[7], seeds[8]), a512 = fr(seeds[4], seeds[11]),
        a611 = fr(seeds[5], seeds[10]), a710 = fr(seeds[6], seeds[9]);
      const q1 = playN(seeds[0], a89), q2 = playN(seeds[3], a512),
        q3 = playN(seeds[2], a611), q4 = playN(seeds[1], a710);
      for (const t of [seeds[0], seeds[3], seeds[2], seeds[1], a89, a512, a611, a710]) acc.qf[t]++;
      const sf1 = playN(q1, q2), sf2 = playN(q4, q3);
      for (const t of [q1, q2, q3, q4]) acc.sf[t]++;
      const champ = playN(sf1, sf2);
      for (const t of [sf1, sf2]) acc.final[t]++;
      acc.champ[champ]++;
      for (let t = 0; t < n; t++) { acc.winTot[t] += w[t]; acc.gpTot[t] += gp[t]; }
    }
  }

  async function simulatePlayoff(nSims, onProgress) {
    const ctx = buildSimContext();
    const n = ctx.n;
    ctx.homeP = (hi, lo) => ctx.PhArr[hi * n + lo];
    ctx.PhArr = (function () {                       // home matrix (rebuild once)
      const vecs = ctx.ns.map(effVec), a = new Float32Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j)
        a[i * n + j] = winpFromDiff(diffVec(vecs[i], vecs[j]), 1);
      return a;
    })();
    const acc = {};
    for (const k of ["confChamp", "playoff", "bye", "qf", "sf", "final", "champ", "winTot", "gpTot"])
      acc[k] = new Float64Array(n);
    const batch = 1000;
    for (let done = 0; done < nSims; done += batch) {
      const b = Math.min(batch, nSims - done);
      runSimChunk(ctx, b, acc);
      if (onProgress) onProgress((done + b) / nSims);
      // Yield via a macrotask (not rAF, which pauses in a backgrounded tab).
      await new Promise(r => setTimeout(r, 0));
    }
    const P = cur().playoff; // reuse conference labels
    const teams = [];
    for (let i = 0; i < n; i++) {
      if (!acc.playoff[i] && !acc.confChamp[i]) continue;
      teams.push({
        team: ctx.ns[i], conference: conf(ctx.ns[i]),
        avg_wins: acc.winTot[i] / nSims, avg_losses: (acc.gpTot[i] - acc.winTot[i]) / nSims,
        conf_champ: acc.confChamp[i] / nSims, playoff: acc.playoff[i] / nSims,
        bye: acc.bye[i] / nSims, qf: acc.qf[i] / nSims, sf: acc.sf[i] / nSims,
        final: acc.final[i] / nSims, champ: acc.champ[i] / nSims,
      });
    }
    teams.sort((a, b) => (b.playoff - a.playoff) || (b.champ - a.champ));
    return { n_sims: nSims, rules: P.rules, teams, custom: true };
  }

  document.getElementById("resim-btn").addEventListener("click", async () => {
    const btn = document.getElementById("resim-btn");
    const status = document.getElementById("resim-status");
    const nSims = +document.getElementById("resim-n").value;
    btn.disabled = true;
    status.className = "resim-status";
    const res = await simulatePlayoff(nSims, p => {
      status.textContent = `Simulating… ${Math.round(100 * p)}%`;
    });
    simData[lens] = res;
    renderPlayoff();
    renderTop25();
    status.className = "resim-status custom";
    status.textContent = `Updated with your inputs (${nSims.toLocaleString()} sims).`;
    btn.disabled = false;
  });

  /* ---------- Matchup simulator ---------- */
  const allNames = rankNames().slice().sort();
  const byConf = {};
  allNames.forEach(t => (byConf[conf(t)] = byConf[conf(t)] || []).push(t));
  function makePicker(el, initial) {
    const sel = document.createElement("select");
    sel.innerHTML = Object.keys(byConf).sort().map(c =>
      `<optgroup label="${esc(c)}">` +
      byConf[c].map(t => `<option value="${esc(t)}" ${t === initial ? "selected" : ""}>${esc(t)}</option>`).join("") +
      `</optgroup>`).join("");
    el.appendChild(sel);
    return sel;
  }
  const selA = makePicker(document.getElementById("pickerA"), "Ohio State");
  const selB = makePicker(document.getElementById("pickerB"), "Notre Dame");

  function predict(a, b, venue) {
    const A = effVec(a), B = effVec(b), M = cur().model;
    const homeA = venue === "A" ? 1 : 0, homeB = venue === "B" ? 1 : 0;
    let pA, marginA;
    if (homeB) {
      const xb = diffVec(B, A);
      const pB = winpFromDiff(xb, 1);
      pA = 1 - pB;
      marginA = -(M.margin.intercept + dot(M.margin.coef, xb) + M.margin.hfa);
    } else {
      const x = diffVec(A, B);
      pA = winpFromDiff(x, homeA);
      marginA = M.margin.intercept + dot(M.margin.coef, x) + homeA * M.margin.hfa;
    }
    const Pt = M.points;
    const ptsA = Pt.intercept + Pt.coef[0] * A[0] + Pt.coef[1] * B[1] + Pt.coef[2] * homeA;
    const ptsB = Pt.intercept + Pt.coef[0] * B[0] + Pt.coef[1] * A[1] + Pt.coef[2] * homeB;
    const total = ptsA + ptsB;
    return { pA, margin: marginA, total, scoreA: (total + marginA) / 2, scoreB: (total - marginA) / 2 };
  }
  function renderMatchup() {
    const a = selA.value, b = selB.value;
    document.getElementById("venueA-label").textContent = "At " + (meta[a]?.abbreviation || a);
    document.getElementById("venueB-label").textContent = "At " + (meta[b]?.abbreviation || b);
    const venue = document.querySelector("input[name=venue]:checked").value;
    if (a === b) {
      document.getElementById("matchup-result").innerHTML =
        `<p class="sub" style="text-align:center">Pick two different teams 🙂</p>`;
      return;
    }
    const r = predict(a, b, venue);
    const fav = r.margin >= 0 ? a : b, spread = Math.abs(r.margin);
    const venueNote = venue === "N" ? "Neutral field" : "At " + (venue === "A" ? a : b);
    document.getElementById("matchup-result").innerHTML = `
      <div class="face-off">
        <div class="side">
          <img src="${logoURL(a)}" alt=""><div class="name">${esc(a)}</div>
          <div class="conf">${esc(conf(a))}</div>
          <div class="winp" style="color:${color(a)}">${pct(r.pA)}</div>
        </div>
        <div class="mid">
          <div class="score">${Math.round(r.scoreA)} · ${Math.round(r.scoreB)}</div>
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
        <span><b>${esc(meta[fav]?.abbreviation || fav)} −${spread.toFixed(1)}</b> spread</span>
        <span><b>${r.total.toFixed(1)}</b> total (O/U)</span>
        <span><b>${Math.round(r.scoreA)}–${Math.round(r.scoreB)}</b> most likely score</span>
      </div>`;
  }
  selA.addEventListener("change", renderMatchup);
  selB.addEventListener("change", renderMatchup);
  document.querySelectorAll("input[name=venue]").forEach(x => x.addEventListener("change", renderMatchup));

  /* ---------- Ratings Lab (editable inputs) ---------- */
  let labOrder = null;                        // cached row order; null = by current rating
  const rowRefs = {};                         // team -> {rankCell, powerCell, inputs[]}
  function renderLab(reorder) {
    const order = labOrder || Object.keys(power).sort((a, b) => power[b] - power[a]);
    if (reorder) labOrder = order.slice();
    const head = `<tr>
      <th>#</th><th>Team</th>
      <th class="grp">Off</th><th class="grp">Def</th><th class="grp">FieldPos</th><th class="grp">Pythag</th>
      <th class="grp roster">Talent</th><th class="grp roster">Returning</th>
      <th class="num">Power</th><th></th></tr>`;
    const body = order.map(t => {
      const v = effVec(t);
      const cells = FIELDS.map(f => `<td><input class="lab-input ${f.grp}${isEdited(t, f.i) ? " edited" : ""}"
        type="number" step="0.05" data-team="${esc(t)}" data-field="${f.i}"
        value="${v[f.i].toFixed(2)}"></td>`).join("");
      return `<tr data-team="${esc(t)}">
        <td class="rank">${rankOf[t]}</td>
        <td><div class="team-cell"><img src="${logoURL(t)}" alt="">
          <div><div>${esc(t)}</div><div class="conf">${esc(conf(t))}</div></div></div></td>
        ${cells}
        <td class="power num">${(100 * power[t]).toFixed(1)}</td>
        <td class="rowtools">${apiAvailable
          ? `<button class="roster-btn ${rosterEdits[lens][t] ? "active" : ""}" data-roster="${esc(t)}" title="Edit ${esc(t)} roster">📋</button>`
          : ""}<button class="row-reset ${teamEditCount(t) ? "active" : ""}" data-reset="${esc(t)}" title="Reset ${esc(t)}">↺</button></td>
      </tr>`;
    }).join("");
    document.getElementById("lab-table").innerHTML =
      `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
    updateEditCount();
  }
  function updateLabCells() {
    // Update only rank + power cells (no re-render) so inputs keep focus.
    document.querySelectorAll("#lab-table tbody tr").forEach(tr => {
      const t = tr.dataset.team;
      tr.querySelector(".rank").textContent = rankOf[t];
      tr.querySelector(".power").textContent = (100 * power[t]).toFixed(1);
    });
  }
  function updateEditCount() {
    const nT = Object.values(overrides[lens]).filter(o => Object.keys(o).length).length;
    const el = document.getElementById("lab-editcount");
    el.textContent = totalEdits() ? `${totalEdits()} edit(s) · ${nT} team(s)` : "";
  }

  let labTimer = null;
  document.getElementById("lab-table").addEventListener("input", e => {
    const inp = e.target;
    if (!inp.classList.contains("lab-input")) return;
    const t = inp.dataset.team, i = +inp.dataset.field;
    const val = parseFloat(inp.value);
    const base = baseVec(t)[i];
    overrides[lens][t] = overrides[lens][t] || {};
    if (isNaN(val) || Math.abs(val - base) < 1e-9) { delete overrides[lens][t][i]; inp.classList.remove("edited"); }
    else { overrides[lens][t][i] = val; inp.classList.add("edited"); }
    if (!Object.keys(overrides[lens][t]).length) delete overrides[lens][t];
    const rr = inp.closest("tr").querySelector(".row-reset");
    rr.classList.toggle("active", teamEditCount(t) > 0);
    saveOverrides();
    clearTimeout(labTimer);
    labTimer = setTimeout(() => {
      computePower(); updateLabCells(); updateEditCount();
    }, 120);
  });
  document.getElementById("lab-table").addEventListener("click", e => {
    const ds = e.target.dataset || {};
    if (ds.roster) { openRoster(ds.roster); return; }
    if (!ds.reset) return;
    delete overrides[lens][ds.reset]; saveOverrides();
    computePower(); renderLab(); // full re-render to reset that row's input values
  });
  document.getElementById("lab-reset").addEventListener("click", () => {
    if (!totalEdits()) return;
    if (!confirm("Reset all input edits for this lens back to the model defaults?")) return;
    overrides[lens] = {}; saveOverrides();
    labOrder = null; computePower(); renderLab(); renderAll();
  });
  document.getElementById("lab-sort").addEventListener("click", () => { labOrder = null; renderLab(true); });
  document.getElementById("lab-search").addEventListener("input", e => {
    const q = e.target.value.toLowerCase();
    document.querySelectorAll("#lab-table tbody tr").forEach(tr =>
      tr.style.display = tr.dataset.team.toLowerCase().includes(q) ? "" : "none");
  });

  /* ---------- Roster editor modal ---------- */
  const RM_GROUPS = ["QB", "RB", "WR", "TE", "OL", "DT", "EDGE", "LB", "CB", "SAF"];
  const modal = document.getElementById("roster-modal");
  let rmTeam = null, rmRoster = [];

  const cloneRoster = t => JSON.parse(JSON.stringify(
    (rosterEdits[lens][t] || ROSTERS[t] || [])));
  const rosterEqual = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  function openRoster(team) {
    if (!ROSTERS) return;
    rmTeam = team;
    rmRoster = cloneRoster(team);
    // display order: by position group, starters first
    rmRoster.sort((a, b) => (RM_GROUPS.indexOf(a.group) - RM_GROUPS.indexOf(b.group))
      || ((a.depth || 2) - (b.depth || 2)) || a.name.localeCompare(b.name));
    document.getElementById("rm-team").textContent = team;
    document.getElementById("rm-sub").textContent =
      `current talent ${effVec(team)[4].toFixed(2)} · rank #${rankOf[team]} · power ${(100 * power[team]).toFixed(1)}`;
    document.getElementById("rm-status").textContent = "";
    document.getElementById("rm-status").className = "rm-status";
    renderModalBody();
    modal.hidden = false;
  }
  function renderModalBody() {
    const orig = ROSTERS[rmTeam] || [];
    const gradeMap = {};                          // original grade per name for "edited" mark
    orig.forEach(p => gradeMap[p.name] = p.grade);
    const rows = rmRoster.map((p, i) => {
      const edited = gradeMap[p.name] !== undefined && gradeMap[p.name] !== p.grade;
      return `<div class="rm-row" data-idx="${i}">
        <input class="pname" value="${esc(p.name)}" data-k="name" placeholder="Player">
        <select data-k="group">${RM_GROUPS.map(g =>
          `<option ${g === p.group ? "selected" : ""}>${g}</option>`).join("")}</select>
        <select data-k="depth">
          <option value="1" ${(+p.depth) === 1 ? "selected" : ""}>Starter</option>
          <option value="2" ${(+p.depth) !== 1 ? "selected" : ""}>Backup</option></select>
        <input class="grade ${edited ? "edited" : ""}" type="number" step="0.5" min="0" max="100"
          data-k="grade" value="${p.grade == null ? "" : p.grade}" placeholder="—">
        <button class="rm-del" data-del="${i}" title="Remove">✕</button>
      </div>`;
    }).join("");
    document.getElementById("rm-body").innerHTML = rows ||
      `<div class="rm-note">No players — add some, or reset.</div>`;
  }
  document.getElementById("rm-body").addEventListener("input", e => {
    const row = e.target.closest(".rm-row"); if (!row) return;
    const i = +row.dataset.idx, k = e.target.dataset.k;
    if (k === "grade") rmRoster[i].grade = e.target.value === "" ? null : parseFloat(e.target.value);
    else if (k === "depth") rmRoster[i].depth = +e.target.value;
    else rmRoster[i][k] = e.target.value;
  });
  document.getElementById("rm-body").addEventListener("click", e => {
    const d = e.target.dataset.del; if (d === undefined) return;
    rmRoster.splice(+d, 1); renderModalBody();
  });
  document.getElementById("rm-add").addEventListener("click", () => {
    rmRoster.push({ name: "New Player", group: "QB", depth: 2, grade: 70 });
    renderModalBody();
    document.querySelector("#rm-body .rm-row:last-child .pname").focus();
  });
  document.getElementById("rm-reset").addEventListener("click", () => {
    rmRoster = JSON.parse(JSON.stringify(ROSTERS[rmTeam] || []));
    rmRoster.sort((a, b) => (RM_GROUPS.indexOf(a.group) - RM_GROUPS.indexOf(b.group))
      || ((a.depth || 2) - (b.depth || 2)));
    renderModalBody();
  });
  const closeModal = () => { modal.hidden = true; rmTeam = null; };
  document.getElementById("rm-close").addEventListener("click", closeModal);
  modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
  document.getElementById("rm-apply").addEventListener("click", async () => {
    const status = document.getElementById("rm-status");
    if (rosterEqual(rmRoster, ROSTERS[rmTeam] || [])) delete rosterEdits[lens][rmTeam];
    else rosterEdits[lens][rmTeam] = rmRoster;
    saveRosterEdits();
    status.className = "rm-status"; status.textContent = "Recomputing…";
    try {
      await postRecompute();
      computePower(); labOrder = null; renderLab(); renderTop25();
      // playoff/matchup re-render on tab open; nudge matchup if built
      status.className = "rm-status done"; status.textContent = "Applied.";
      setTimeout(closeModal, 350);
    } catch (err) {
      status.className = "rm-status"; status.textContent = "Failed: " + err.message;
    }
  });

  /* ---------- boot ---------- */
  function renderAll() { renderTop25(); renderPlayoff(); renderMatchup(); }
  async function boot() {
    try {
      const r = await fetch("/api/rosters");
      if (r.ok) { ROSTERS = await r.json(); apiAvailable = true; }
    } catch (e) { /* static server: roster editing off */ }
    const hint = document.getElementById("roster-hint");
    if (apiAvailable) {
      hint.className = "roster-hint";
      hint.innerHTML = "📋 <b>Roster editing is ON</b> — click the clipboard on any " +
        "team to edit its 2026 depth chart and 2025 grades; talent re-derives on the backend.";
      // re-apply any persisted roster edits (both lenses)
      for (const L of ["", "_roster"]) {
        if (Object.keys(rosterEdits[L]).length) {
          const saved = lens; lens = L;
          try { await postRecompute(); } catch (e) { console.warn("roster re-apply failed", e); }
          lens = saved;
        }
      }
    } else {
      hint.className = "roster-hint off";
      hint.innerHTML = "Player-level roster editing needs the local Python server " +
        "(<code>./venv/bin/python -m scripts.serve</code>). Served statically now — the " +
        "six inputs above are still directly editable.";
    }
    computePower(); renderAll(); renderLab();
  }
  boot();
})();
