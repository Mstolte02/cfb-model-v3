/* CFB Model v3 — 2026 visuals.
   Tabs: Ratings · Playoff Projection · Matchup Simulator · Team Breakdown.
   One rating variant; the talent blend is read from diagnostics.json.

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
  const [teams, schedule, players, diag, ratings, playoff, model] = await Promise.all([
    fetchJSON("data/teams.json"),
    fetchJSON("data/schedule.json"),
    fetchJSON("data/players.json").catch(() => ({})),
    fetchJSON("data/diagnostics.json").catch(() => null),
    fetchJSON("data/ratings.json"),
    fetchJSON("data/playoff.json"),
    fetchJSON("data/model.json"),
  ]);
  // One model, one set of numbers. The old lens toggle offered a second
  // roster-weighted variant that leaned harder on the two-deep; it was a knowingly
  // worse backtest kept as an alternative view, and it is gone. Talent is the
  // a PFF / recruiting / WAR blend whose weights are swept jointly under
  // leave-one-season-out and exported, not written in here.
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

  const vecOf = t => cur().model.teams[t];
  const rankNames = () => cur().ratings.teams.map(t => t.team);
  const ratingRow = () => Object.fromEntries(cur().ratings.teams.map(t => [t.team, t]));
  const simRow = () => Object.fromEntries((cur().playoff.teams || []).map(t => [t.team, t]));

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

  /* Rounded display scores. Rounding each side independently can tie a game the
     model does not think is a tie - a 0.4-point projected margin lands both sides
     on the same integer - so the favourite takes the extra point. Football has no
     ties, and a bracket that shows one reads as a bug. */
  /* The winner is whoever the ensemble win probability favours, and the displayed
     score has to agree with it. Those two can diverge on a near-coin-flip, because
     pA blends the logistic with the margin model while the score comes from the
     margin alone: a 51% favourite can carry a margin of -0.2. Left alone that put a
     team in the championship game after losing its semifinal on the scoreboard. */
  function displayScore(r, winnerIsA) {
    let a = Math.max(0, Math.round(r.scoreA)), b = Math.max(0, Math.round(r.scoreB));
    if (winnerIsA === undefined) winnerIsA = r.pA >= 0.5;
    if (winnerIsA && a <= b) a = b + 1;
    else if (!winnerIsA && b <= a) b = a + 1;
    return [a, b];
  }

  /* A margin under half a point is a coin flip, and "−0.0" reads as a rendering
     bug rather than a pick'em. */
  function spreadLabel(margin) {
    if (Math.abs(margin) < 0.5) return `<span class="pk">PK</span>`;
    // en dash rather than the minus sign, which sets far too wide next to figures
    const sign = margin >= 0 ? "–" : "+";
    return `<span class="spread ${margin >= 0 ? "pos" : "neg"}"><span class="sgn">${sign}</span>${Math.abs(margin).toFixed(1)}</span>`;
  }

  /* Diverging bar for a z-scored input: the midline is FBS average. */
  function zBar(v, tint, span = 2.2) {
    const f = Math.max(-1, Math.min(1, v / span));
    const w = Math.abs(f) * 50;
    const left = f >= 0 ? 50 : 50 - w;
    return `<div class="zbar"><span class="zmid"></span>
      <i style="left:${left}%;width:${w}%;background:${tint}"></i></div>`;
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
  const COLS = [
    { k: "rank",       h: "#",            n: true },
    { k: "team",       h: "Team",         n: false },
    { k: "power",      h: "Power",        n: true },
    { k: "record",     h: "Proj Record",  n: true, sort: r => r.avg_wins ?? -1,
      fmt: r => r.avg_wins != null ? `${r.avg_wins.toFixed(1)}–${r.avg_losses.toFixed(1)}` : "—" },
    { k: "O",          h: "Offense",      n: true },
    { k: "D",          h: "Defense",      n: true },
    { k: "talent",     h: "Talent",       n: true },
    { k: "returning",  h: "Returning",    n: true },
    { k: "sos",        h: "SOS",          n: true,
      fmt: r => (r.sos >= 0 ? "+" : "") + r.sos.toFixed(2) },
    { k: "conf_champ", h: "Conf %",       n: true, fmt: r => pct(r.conf_champ ?? 0, 0) },
    { k: "playoff",    h: "CFP %",        n: true, fmt: r => pct(r.playoff ?? 0, 0) },
    { k: "champ",      h: "Natty %",      n: true, fmt: r => pct(r.champ ?? 0, 1) },
  ];
  let sortKey = "power", sortDesc = true;

  function dashRows() {
    const q = document.getElementById("dash-search").value.trim().toLowerCase();
    const c = document.getElementById("dash-conf").value;
    const tier = document.getElementById("dash-tier").value;
    let rows = cur().ratings.teams.slice();
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
    const all = cur().ratings.teams;
    const maxPower = Math.max(...all.map(r => r.power));
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
        if (c.k === "power") return `<td><div class="bar-wrap"><div class="bar">
            <i style="width:${100 * r.power / maxPower}%;background:${tint}"></i>
          </div><span class="pct">${(100 * r.power).toFixed(1)}</span></div></td>`;
        if (["O", "D", "talent", "returning"].includes(c.k)) {
          const v = r[c.k];
          return `<td class="num"><div class="zcell">${zBar(v, tint)}
            <span>${v >= 0 ? "+" : ""}${v.toFixed(2)}</span></div></td>`;
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
      // Resolve forward: score each game, advance the projected winner into the
      // slot it feeds. feeds maps a game index to the games that supply its teams.
      const G = br.games.map(g => ({ ...g }));
      const feeds = br.feeds || {};
      for (let i = 0; i < G.length; i++) {
        const g = G[i];
        const src = feeds[String(i)];
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

      const seedOf = {};
      br.seeds.forEach((t, i) => { if (t) seedOf[t] = i + 1; });
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
      document.getElementById("bracket").innerHTML = `
        <div class="round-label">First Round</div><div class="round-label">Quarterfinals</div>
        <div class="round-label">Semifinals</div><div class="round-label">Championship</div>
        <div class="round-col">${G.slice(0, 4).map(card).join("")}</div>
        <div class="round-col">${G.slice(4, 8).map(card).join("")}</div>
        <div class="round-col">${G.slice(8, 10).map(card).join("")}</div>
        <div class="round-col">${card(G[10])}
          ${champ ? `<div class="trophy-card" style="--wash:${rgba(champ, .18)}">
            <div class="emoji">🏆</div><img src="${logoURL(champ)}" alt="">
            <div class="champ-name" style="color:${color(champ)}">${esc(champ)}</div>
            <div class="bye-tag">projected champion ·
              ${pct((byTeam[champ] || {}).champ || 0, 1)} title odds</div></div>` : ""}
        </div>`;
    }

    renderSelection(P, byTeam);

    const maxCFP = Math.max(...P.map(t => t.playoff));
    document.getElementById("playoff-table").innerHTML = `
      <table><thead><tr>
        <th></th><th>Team</th><th class="num">Proj Record</th><th>Conf&nbsp;Champ</th>
        <th>Make&nbsp;CFP</th><th class="num">Bye</th><th class="num">Semis</th>
        <th class="num">Final</th><th class="num">Natty</th>
      </tr></thead><tbody>${P.slice(0, 40).map((t, i) => `
        <tr>
          <td class="rank">${i + 1}</td>
          <td><div class="team-cell">
            <span class="team-stripe" style="background:${color(t.team)}"></span>
            <img src="${logoURL(t.team)}" alt="" loading="lazy">
            <div><button class="team-link" data-team="${esc(t.team)}">${esc(t.team)}</button>
              <div class="conf">${esc(t.conference)}</div></div></div></td>
          <td class="num">${t.avg_wins.toFixed(1)}–${t.avg_losses.toFixed(1)}</td>
          <td><div class="bar-wrap"><div class="bar"><i class="green" style="width:${100 * t.conf_champ}%"></i></div>
            <span class="pct">${pct(t.conf_champ)}</span></div></td>
          <td><div class="bar-wrap"><div class="bar"><i style="width:${100 * t.playoff / maxCFP}%;background:${color(t.team)}"></i></div>
            <span class="pct">${pct(t.playoff)}</span></div></td>
          <td class="num">${pct(t.bye)}</td>
          <td class="num">${pct(t.sf)}</td>
          <td class="num">${pct(t.final)}</td>
          <td class="num"><b>${pct(t.champ)}</b></td>
        </tr>`).join("")}</tbody></table>`;
    wireTeamLinks();
  }

  const P4_CONFS = ["ACC", "Big 12", "Big Ten", "SEC"];
  const G6_CONFS = ["American Athletic", "Conference USA", "Mid-American",
                    "Mountain West", "Pac-12", "Sun Belt"];

  /* How the twelve teams are chosen, and who is in line for the five automatic bids.
     The bracket above shows an outcome; this shows the rule that produced it. Worth
     spelling out because the 2026 format changed - the Group of 6 bid is no longer
     tied to winning a conference - and because the ranking underneath is a fitted
     model of the committee rather than the model's own opinion of who is best. */
  function renderSelection(P, byTeam) {
    const el = document.getElementById("playoff-explain");
    const pl = cur().playoff;
    const cw = pl.committee_weights;
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
            <b>schedule strength</b>, and much less about how good a team looks.</p></div>
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
    document.getElementById("matchup-result").innerHTML = `
      <div class="face-off">
        <div class="side">
          <img src="${logoURL(a)}" alt=""><div class="name">${esc(a)}</div>
          <div class="conf">${esc(conf(a))}</div>
          <div class="winp" style="color:${color(a)}">${pct(r.pA)}</div>
        </div>
        <div class="mid">
          <div class="score">${displayScore(r)[0]} · ${displayScore(r)[1]}</div>
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
        <span><b>${displayScore(r).join("–")}</b> most likely score</span>
      </div>`;
  }
  selA.addEventListener("change", renderMatchup);
  selB.addEventListener("change", renderMatchup);
  document.querySelectorAll("input[name=venue]").forEach(x =>
    x.addEventListener("change", renderMatchup));

  /* =======================================================================
     TEAM BREAKDOWN
     ======================================================================= */
  const GROUP_ORDER = ["QB", "RB", "WR", "TE", "OL", "DT", "EDGE", "LB", "CB", "SAF"];
  const OFF_GROUPS = new Set(["QB", "RB", "WR", "TE", "OL"]);
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

  function shortName(n) {
    const parts = String(n).trim().split(/\s+/);
    return parts.length < 2 ? n : `${parts[0][0]}. ${parts.slice(1).join(" ")}`;
  }

  function placeSide(players, tint, groups) {
    const html = [];
    const extent = {};                    // group -> [leftmost x, rightmost x]
    for (const g of groups) {
      const lvl = LEVELS[g];
      const members = players.filter(p => p.g === g);
      if (!members.length || !lvl) continue;
      // left-ish labels first, slot/nickel last, so sides come out where expected
      members.sort((a, b) => sideKey(a) - sideKey(b));
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
            ? ` <span class="tag imp" title="No prior FBS snaps">·</span>` : ""}</div>
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
        const b = budget[p.g];
        const n = (seen[p.g] = (seen[p.g] || 0) + 1);
        if (!b) return { p, r: 60 + n };
        const [cap, base, over] = b;
        return { p, r: n <= cap ? base + (n - 1) : over + n * 0.01 };
      })
      .sort((a, b) => a.r - b.r)
      .slice(0, 11)
      .map(x => x.p);
  }

  function lineupHTML(team, roster, tint) {
    const starters = roster.players.filter(p => p.d === 1);
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
  function winsLedger(roster, S, tint) {
    if (roster.projWins == null) return "";
    const rows = [
      ["Replacement-level team", roster.replWins, "var(--muted)"],
      ["Roster value above replacement", roster.winsTotal, tint],
      ["Schedule &amp; context", roster.context, roster.context >= 0 ? "var(--green)" : "var(--red)"],
    ];
    const span = Math.max(...rows.map(r => Math.abs(r[1])), 1);
    return `<div class="ledger">
      ${rows.map(([label, v, c]) => `
        <div class="gr-row">
          <span class="gr-name" style="width:200px">${label}</span>
          <div class="gr-bar"><i style="width:${100 * Math.abs(v) / span}%;
            background:${c}"></i></div>
          <span class="gr-val">${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}</span>
        </div>`).join("")}
      <div class="gr-row ledger-total">
        <span class="gr-name" style="width:200px"><b>Projected wins</b></span>
        <div class="gr-bar"></div>
        <span class="gr-val"><b>${roster.projWins.toFixed(2)}</b></span>
      </div></div>`;
  }

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

    /* ---- model inputs ---- */
    const inputs = [
      ["Offense", R.O], ["Defense", R.D], ["Talent", R.talent],
      ["Returning", R.returning], ["Pythagorean", R.pythag], ["Schedule (SOS)", R.sos],
    ].filter(x => x[1] != null);
    const inputHTML = inputs.map(([label, v]) => `
      <div class="gr-row">
        <span class="gr-name">${label}</span>
        ${zBar(v, tint)}
        <span class="gr-val">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>
      </div>`).join("");

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
        <td><div class="bar-wrap"><div class="bar">
          <i style="width:${100 * (r ? r.pA : FCS_WIN_P)}%;background:${win ? tint : "var(--red)"}"></i>
        </div><span class="pct">${r ? pct(r.pA, 0) : "95%"}</span></div></td>
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
        ${roster ? winsLedger(roster, S, tint) : ""}
        <div class="wd-foot">Projected starters from the 2026 two-deep, each carrying
          the wins he adds over a replacement-level player. Raw WAR is compressed
          &mdash; the rating underneath is a noisy estimate and a projection is a
          conditional mean &mdash; so every player is scaled by one league-wide
          factor (&times;1.64) that restores the historical spread without moving the
          league average. <span class="tag imp">·</span> marks a player with no prior
          FBS snaps.</div></div>
      ${tossupHTML(t, tint, S.playoff)}
      <div class="panel"><h3>Model inputs</h3>
        <div class="gr wide">${inputHTML}</div>
        <div class="wd-foot">Each input on the model's z-scale, where
          <b>0 = FBS average</b> and higher is better (defense already flipped).</div>
      </div>
      <div class="panel"><h3>2026 schedule
        <span class="hint">— projected score, spread and win probability for every game</span></h3>
        <div class="mini-wrap"><table class="mini sched"><thead><tr>
          <th class="num">Wk</th><th class="num">Date</th><th></th><th>Opponent</th>
          <th class="num">Proj score</th><th class="num">Spread</th><th>Win prob</th>
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

  /* =======================================================================
     METHOD — what goes in, how much it overlaps, what came out
     ======================================================================= */
  const ACC = "var(--accent)", GRN = "var(--green)", RED = "var(--red)";

  /* A correlation cell: sign by hue, magnitude by ink. Diagonal is muted so the
     eye goes to the off-diagonal, which is the part that matters. */
  function corrCell(v, self) {
    if (v == null) return `<td class="cc"></td>`;
    const a = Math.min(1, Math.abs(v));
    const bg = self ? "transparent"
      : v >= 0 ? `rgba(47,96,150,${(a * 0.75).toFixed(2)})`
               : `rgba(168,50,38,${(a * 0.75).toFixed(2)})`;
    const strong = !self && a >= 0.6;
    return `<td class="cc${self ? " self" : ""}${strong ? " strong" : ""}"
      style="background:${bg}">${v.toFixed(2)}</td>`;
  }
  function corrTable(names, corr, note) {
    const head = `<tr><th></th>${names.map(n => `<th class="num">${esc(n)}</th>`).join("")}</tr>`;
    const rows = names.map(a => `<tr><th class="rowh">${esc(a)}</th>${
      names.map(b => corrCell(corr[a] ? corr[a][b] : null, a === b)).join("")}</tr>`).join("");
    return `<div class="mini-wrap"><table class="corr">${head}${rows}</table></div>
      ${note ? `<div class="wd-foot">${note}</div>` : ""}`;
  }

  /* Horizontal bar for a metric where lower is better (Brier). */
  function metricBar(v, lo, hi, tint) {
    const f = Math.max(0, Math.min(1, (hi - v) / (hi - lo || 1)));
    return `<div class="gr-bar"><i style="width:${(f * 100).toFixed(1)}%;
      background:${tint}"></i></div>`;
  }

  function renderMethod() {
    const el = document.getElementById("method-body");
    if (!diag) {
      el.innerHTML = `<p class="sub">No diagnostics found — run
        <code>scripts.export_diagnostics</code>.</p>`;
      return;
    }
    const D = diag;

    // ---- 1. sources -------------------------------------------------------
    const srcRows = (D.sources || []).map(s => `
      <tr>
        <td><b>${esc(s.name)}</b><div class="conf">${esc(s.provider)}</div></td>
        <td><span class="tag">${esc(s.feeds)}</span></td>
        <td class="small" style="white-space:normal">${esc(s.detail)}</td>
        <td class="num small">${esc(s.years)}</td>
      </tr>`).join("");

    // ---- 2. talent sources ------------------------------------------------
    const ts = D.talent_sources || {};
    const tsw = ts.weights || {};
    const wRow = ["PFF", "CFBD", "WAR"].map(n => `
      <div class="wsplit-seg" style="width:${((tsw[n] || 0) * 100).toFixed(1)}%;
        background:${n === "WAR" ? ACC : n === "PFF" ? "var(--blue)" : GRN}">
        <span>${esc(n)} ${Math.round((tsw[n] || 0) * 100)}%</span></div>`).join("");
    const contrasts = ((D.talent_sweep || {}).contrasts) || [];
    const cb = contrasts.map(c => c.brier);
    const clo = Math.min(...cb, 0.204), chi = Math.max(...cb, 0.209);
    const contrastRows = contrasts.map(c => `
      <div class="gr-row">
        <span class="gr-name" style="width:230px">${esc(c.label)}</span>
        ${metricBar(c.brier, clo, chi, c.label.includes("best") ? ACC : "var(--line2)")}
        <span class="gr-val">${c.brier.toFixed(4)}</span>
      </div>`).join("");

    // ---- 3. features ------------------------------------------------------
    const F = D.features || {};
    const fnames = F.names || [];
    const vifRows = fnames.map(f => {
      const v = (F.vif || {})[f];
      const dropped = (F.dropped || []).includes(f);
      return `<div class="gr-row">
        <span class="gr-name ${dropped ? "" : "off"}" style="width:110px">${esc(f)}${
          dropped ? ` <span class="tag">retired</span>` : ""}</span>
        ${v == null ? `<div class="gr-bar"></div><span class="gr-val">—</span>`
          : `<div class="gr-bar"><i style="width:${Math.min(100, v / 5 * 100)}%;
               background:${v > 5 ? RED : "var(--blue)"}"></i></div>
             <span class="gr-val">${v.toFixed(2)}</span>`}
      </div>`;
    }).join("");

    const co = D.coefficients || {};
    const lg = co.logistic || {};
    const cmax = Math.max(...Object.values(lg).map(Math.abs), 0.01);
    const coefRows = fnames.map(f => {
      const v = lg[f] ?? 0;
      const dropped = (F.dropped || []).includes(f);
      return `<div class="gr-row">
        <span class="gr-name ${dropped ? "" : "off"}" style="width:110px">${esc(f)}</span>
        ${zBar(v, dropped ? "var(--line2)" : ACC, cmax * 1.05)}
        <span class="gr-val">${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}</span>
      </div>`;
    }).join("") + `
      <div class="gr-row">
        <span class="gr-name off" style="width:110px">home field</span>
        ${zBar(co.logistic_hfa ?? 0, GRN, cmax * 1.05)}
        <span class="gr-val">+${(co.logistic_hfa ?? 0).toFixed(3)}</span>
      </div>`;

    // ---- 4. evaluation ----------------------------------------------------
    const ev = D.evaluation || {};
    const ps = ev.per_season || [];
    const seasonRows = ps.map(s => `
      <tr><td>${s.season}</td><td class="num small">${s.n.toLocaleString()}</td>
        <td class="num">${s.brier.toFixed(4)}</td>
        <td class="num">${s.log_loss.toFixed(4)}</td>
        <td class="num">${(100 * s.accuracy).toFixed(1)}%</td></tr>`).join("");
    const cal = ev.calibration || [];
    const calBars = cal.map(c => {
      const h = Math.max(2, c.actual * 100);
      const ph = Math.max(2, c.pred * 100);
      return `<div class="cal-col" title="predicted ${pct(c.pred)}, actual ${pct(c.actual)}, n=${c.n}">
        <div class="cal-pair">
          <div class="cal-bar pred" style="height:${ph}%"></div>
          <div class="cal-bar act" style="height:${h}%"></div>
        </div>
        <span class="wd-x">${c.lo.toFixed(1)}</span></div>`;
    }).join("");

    // ---- 5. WAR internals -------------------------------------------------
    const wf = D.war_facets || [];
    const wsplit = D.war_source_split || {};
    const topFacets = wf.slice(0, 10);
    const fmaxw = Math.max(...topFacets.map(f => f.weight), 0.01);
    const facetRows = topFacets.map(f => `
      <div class="gr-row">
        <span class="gr-name" style="width:130px">${esc(f.facet)}</span>
        <div class="gr-bar"><i style="width:${100 * f.weight / fmaxw}%;
          background:${f.source === "CFBD" ? GRN : "var(--blue)"}"></i></div>
        <span class="gr-val">${(100 * f.weight).toFixed(1)}%</span>
      </div>`).join("");

    // ---- 6. decisions -----------------------------------------------------
    const decRows = (D.decisions || []).map(d => `
      <div class="dec">
        <div class="dec-q">${esc(d.question)}</div>
        <div class="dec-a">${esc(d.answer)}</div>
        <div class="dec-e">${esc(d.evidence)}</div>
      </div>`).join("");

    // The page used to open with a correlation matrix. Most people who look at this
    // want to know what it does and whether to believe it; the collinearity
    // diagnostics answer neither. Plain walkthrough first, accuracy second, the
    // technical detail folded away at the bottom for whoever wants it.
    const acc = ((ev.loso || {}).accuracy || 0) * 100;
    const tsw2 = (D.talent_sources || {}).weights || {};
    const pcw = n => Math.round((tsw2[n] || 0) * 100);

    el.innerHTML = `
      <div class="steps">
        <div class="step"><span class="step-n">1</span>
          <h4>Rate every team</h4>
          <p>Each team gets a handful of numbers: how good its offense and defense
            have been against the schedule they faced, how much talent is on the
            roster, how much of last year's production is coming back, and what its
            scoring margin implied it deserved to win.</p></div>
        <div class="step"><span class="step-n">2</span>
          <h4>Turn ratings into a game</h4>
          <p>For any two teams the model converts the gap between those numbers into a
            win probability and a projected score. It was fitted on
            ${(ev.per_season || []).reduce((s, r) => s + r.n, 0).toLocaleString()}
            real games and it knows what home field is worth.</p></div>
        <div class="step"><span class="step-n">3</span>
          <h4>Play the season 20,000 times</h4>
          <p>The real 2026 schedule is played out over and over, with conference title
            games and the twelve-team playoff bracket, so nothing on this site can
            break a rule the actual sport enforces.</p></div>
        <div class="step"><span class="step-n">4</span>
          <h4>Report the spread, not just the guess</h4>
          <p>Because it is 20,000 seasons rather than one, every number here is a
            range. A team's floor and ceiling are as much of the answer as its
            projected record.</p></div>
      </div>

      <div class="panel"><h3>How accurate is it?</h3>
        <p class="sub">Tested the honest way: the model is retrained with one season
          completely removed, then asked to predict that season's games having never
          seen them. Repeated for every season.</p>
        <div class="od-split">
          <div class="od-chip2" style="--tint:${ACC}">
            <b>${acc.toFixed(1)}%</b><span>of games called correctly</span></div>
          <div class="od-chip2" style="--tint:var(--line2)">
            <b>${pct(ev.home_win_rate ?? 0, 0)}</b><span>always pick the home team</span></div>
          <div class="od-chip2" style="--tint:${ACC}">
            <b>${(ev.loso || {}).brier?.toFixed(3) ?? "—"}</b><span>Brier score</span></div>
          <div class="od-chip2" style="--tint:var(--line2)">
            <b>${(ev.baselines || {}).coin_flip?.toFixed(3) ?? "—"}</b>
            <span>Brier, coin flip</span></div>
        </div>
        <div class="split">
          <div><h4>Is a stated 70% really 70%?</h4>
            <div class="cal">${calBars}</div>
            <div class="cal-key">
              <span><i style="background:var(--line2)"></i> what the model said</span>
              <span><i style="background:${ACC}"></i> what happened</span>
            </div>
            <div class="wd-foot">Games grouped by the probability the model gave them.
              The two bars matching is the thing that matters &mdash; it means the
              percentages on this site can be taken at face value, not just the
              favourites.</div>
          </div>
          <div><h4>In plain terms</h4>
            <p class="sub">A <b>Brier score</b> is the average squared miss on a
              probability, so lower is better and a coin flip scores
              ${(ev.baselines || {}).coin_flip?.toFixed(3) ?? "0.250"}. Getting
              ${acc.toFixed(0)}% of games right sounds modest, and it is: college
              football is genuinely unpredictable, and any model claiming much more is
              either overfitted or being tested on games it has already seen.</p>
          </div>
        </div>
      </div>

      <div class="panel"><h3>What goes in</h3>
        <div class="mini-wrap"><table class="mini"><thead><tr>
          <th>Source</th><th>Feeds</th><th>What it is</th><th class="num">Years</th>
        </tr></thead><tbody>${srcRows}</tbody></table></div>
        <div class="wd-foot">"Talent" is three separate measures of the same roster
          blended together &mdash; PFF's player grades (${pcw("PFF")}%), recruiting
          rankings (${pcw("CFBD")}%), and a custom wins-above-replacement build
          (${pcw("WAR")}%). The mix was chosen by testing every combination, not by
          picking one.</div>
      </div>

      <div class="panel"><h3>Questions we asked, and what the data said</h3>
        <div class="decs">${decRows}</div>
      </div>

      <details class="nerd"><summary>The technical version</summary>
      <div class="panel"><h3>The three talent signals, and how much they overlap</h3>
        <p class="sub">WAR is built from PFF grades, so the obvious worry is that it
          measures the same thing twice. It does overlap — but at
          <b>r = ${((ts.corr || {}).PFF || {}).WAR?.toFixed(2) ?? "—"}</b>, not
          near one. WAR weights by snaps, uses facet weights fitted to wins, adds
          CFBD play value and subtracts a replacement level; the PFF signal is a
          position-weighted grade average. They diverge enough to be worth carrying
          separately.</p>
        <div class="split">
          <div><h4>Correlation between sources</h4>
            ${corrTable(ts.names || [], ts.corr || {},
              `Pooled over ${(ts.seasons_used || []).join(", ")}
               (${ts.n || 0} team-seasons). 2021 is excluded: with no 2020 grades the
               PFF signal falls back to recruiting entirely and correlates at exactly
               1.00, which would overstate the overlap.`)}
          </div>
          <div><h4>Blend actually used</h4>
            <div class="wsplit">${wRow}</div>
            <div class="wd-foot">Chosen by grid-searching all 45 three-way blends
              under leave-one-season-out, not assembled one at a time.</div>
            <h4 style="margin-top:18px">What each combination is worth</h4>
            ${contrastRows}
            <div class="wd-foot">Brier, lower is better. Dropping PFF for WAR costs
              more than dropping WAR for PFF — the two are complements, and PFF is
              the stronger of the pair.</div>
          </div>
        </div>
      </div>

      <div class="panel"><h3>Model features</h3>
        <div class="split">
          <div><h4>Correlation between feature differences</h4>
            ${corrTable(fnames, F.corr || {},
              "Computed on the matchup differences the model actually fits. Retired "
              + "features are zeroed, so their row and column are blank.")}
          </div>
          <div><h4>Variance inflation</h4>
            ${vifRows}
            <div class="wd-foot">Above 5 is usually called a collinearity problem.
              After retiring ${(F.dropped || []).map(esc).join(" and ")}, nothing is
              close.</div>
            <h4 style="margin-top:18px">What the model learned</h4>
            ${coefRows}
            <div class="wd-foot">Logistic coefficients on the z-scale. Retired
              features sit at exactly zero by construction.</div>
          </div>
        </div>
      </div>

      <div class="panel"><h3>Season by season</h3>
        <div class="mini-wrap"><table class="mini"><thead><tr>
          <th>Season</th><th class="num">Games</th><th class="num">Brier</th>
          <th class="num">Log loss</th><th class="num">Acc</th>
        </tr></thead><tbody>${seasonRows}</tbody></table></div>
        <div class="wd-foot">Each row is a season the model never saw during
          training. Home teams win ${pct(ev.home_win_rate ?? 0)} of games; always
          picking them scores
          ${(ev.baselines || {}).home_team_always?.toFixed(4) ?? "—"}.</div>
      </div>

      ${wf.length ? `<div class="panel"><h3>Inside the WAR build</h3>
        <p class="sub">WAR is its own model feeding one input here. ${wf.length ?
          "Ninety-eight" : "Several dozen"} candidate skills — every PFF metric that
          has a snap denominator, crossed with position group, plus CFBD play value —
          are weighted by non-negative ridge fitted to the <i>following</i> season's
          wins, turned into Massey ratings, then converted to wins above replacement.
          Fitting to the next season rather than the same one matters: same-season
          fits reward coverage grade, which is partly an effect of already winning.</p>
        <div class="split">
          <div><h4>Heaviest facets</h4>${facetRows}
            <div class="wd-foot">
              <span class="tag" style="background:rgba(47,96,150,.14);color:var(--blue)">PFF</span>
              ${((wsplit.PFF ?? 0) * 100).toFixed(0)}% of total weight ·
              <span class="tag" style="background:rgba(63,125,58,.16);color:var(--green)">CFBD</span>
              ${((wsplit.CFBD ?? 0) * 100).toFixed(0)}%</div>
          </div>
          <div><h4>Why not CFBD alone</h4>
            <p class="sub">CFBD has no player-level line data at all, and its defensive
              numbers are counting stats with no snap denominator — a corner nobody
              throws at is invisible. PFF covers what CFBD cannot; CFBD adds
              outcome-anchored play value PFF grades do not have. Facets against
              adjusted win percentage: PFF alone 0.826, CFBD alone 0.741,
              both 0.845.</p>
          </div>
        </div>
      </div>` : ""}
      </details>`;
  }

  /* ---------- boot ---------- */
  function render(view) {
    if (view === "dash") renderDash();
    else if (view === "playoff") renderPlayoff();
    else if (view === "matchup") renderMatchup();
    else if (view === "team") renderTeam();
    else if (view === "method") renderMethod();
  }
  function renderAll() {
    fillConfSelect();
    renderDash(); renderPlayoff(); renderMatchup(); renderTeam(); renderMethod();
  }
  renderAll();
})();
