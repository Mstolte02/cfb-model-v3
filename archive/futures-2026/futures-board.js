/* Futures hub code removed from viz/app.js. This file is not loaded by the
   site. Each block says where it went; paste it back in that place. */

/* ---- In the data loader at the top: `futuresPreseason` was the last name in the destructured list, after `deservingModel`, and this fetch was the last entry of the Promise.all array. ---- */
lockedResults, deservingModel, futuresPreseason] = await Promise.all([
    fetchJSON("data/futures_preseason.json").catch(() => ({ heisman_war: {} })),

/* ---- The season forecast table (#dash-table). It sat between openRoute() and the PLAYOFF section banner. ---- */
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


/* ---- The first two rows of BET_RULES, above the `spread` row. ---- */
    futures:   { minModelP: .50, minGap: .05 },
    // Win totals sit against a de-vigged price the model beats almost everywhere,
    // plus the calibration study's own quantity: the model's expected wins must
    // clear the posted line by minWinGap before a row earns the flag.
    win_total: { minModelP: .50, minGap: .20, minWinGap: .5 },

/* ---- The board's model side and renderers. They sat right after BET_RULES. teamMini() sat between renderFutures() and renderOutcomeBands() and is still in app.js, because the Market board and Tracking use it. ---- */
  const quantileFromCounts = (counts, q) => {
    const total = counts.reduce((a, b) => a + b, 0), target = total * q;
    let seen = 0;
    for (let i = 0; i < counts.length; i++) {
      seen += counts[i];
      if (seen >= target) return i;
    }
    return counts.length - 1;
  };
  /* ---------- THE FUTURES BOARD IS A PRESEASON BOARD ----------
     Every price on this tab was posted in July or August and none of these bets can
     be placed now - a national-title ticket is bought before the season, and the
     board exists to record whether the model disagreed with that price AT THE TIME.
     So the model's side of the comparison has to be its preseason number too.

     It was not. The market probabilities are already frozen - `odds.sources` dates
     every futures book to July/August and the weekly snapshot job only refreshes
     `cfbd_lines` - but the model column read the LIVE simulation, and worse, it
     followed whatever the Playoff tab's version toggle was set to, so a control on
     another tab silently changed what this one claimed the edge was. Between the
     preseason run and 18 September 2026 Ohio State's title probability moved .1127
     to .1782 and its playoff probability .7554 to .8239. Against a price from
     10 August that is a manufactured edge: the model is being credited for knowing
     things it learned in the three weeks after the bet.

     preseasonSim and preseasonPredict are therefore the ONLY model source this tab
     reads. Both come from payload fields that already exist and already describe
     themselves as the preseason snapshot - playoff_preseason.json carries
     `locked: true` and `basis: "Preseason model snapshot"`, and
     model_v4.json.dynamic.preseason_ratings holds the week-0 rating for all 138
     teams. Nothing here is recomputed and nothing here moves again. */
  const preseasonSim = () =>
    Object.fromEntries((playoffPreseason.teams || []).map(t => [t.team, t]));
  const preseasonRating = t => ((cur().model.dynamic || {}).preseason_ratings || {})[t];

  /* Week-0 win probability for one matchup, from the week-0 ratings. Mirrors the
     dynamic half of winpTeams; the futures board never used the static blend, and
     the shipped blend is 1.0, so the dynamic rating IS the model here. */
  function preseasonPredict(home, away, neutral) {
    const rh = preseasonRating(home), ra = preseasonRating(away);
    if (rh == null || ra == null) return null;
    return sigmoid(rh - ra + (neutral ? 0 : cur().model.logistic.hfa));
  }

  function regularWinDist(team) {
    let dist = [1];
    for (const g of schedule) {
      if (g.h !== team && g.a !== team) continue;
      let p;
      const r = preseasonPredict(g.h, g.a, !!g.n);
      if (r != null) {
        p = g.h === team ? r : 1 - r;
      } else p = vecOf(team) ? FCS_WIN_P : 1 - FCS_WIN_P;
      const next = Array(dist.length + 1).fill(0);
      dist.forEach((v, i) => { next[i] += v * (1 - p); next[i + 1] += v * p; });
      dist = next;
    }
    return dist;
  }

  let futureMarket = "national_title";
  function marketSource(market, book) {
    const key = market === "national_title" ? (book === "BetMGM" ? "betmgm_title" : "fanduel_title")
      : market === "make_cfp" ? "fanduel_cfp"
      : market === "conference_title" ? "draftkings_conference"
      : market === "win_totals" ? "betmgm_wins" : "betmgm_heisman";
    return odds.sources && odds.sources[key];
  }
  function setFutureBooks() {
    const sel = document.getElementById("future-book");
    const books = Object.keys((odds.markets || {})[futureMarket] || {});
    const old = sel.value;
    sel.innerHTML = books.map(b => `<option${b === old ? " selected" : ""}>${esc(b)}</option>`).join("");
  }
  function heismanIndex(rows) {
    // Preseason on both inputs, for the same reason as the rest of the tab: the
    // Heisman price is an August price. avg_wins comes from the preseason
    // simulation rather than ratings.json, whose avg_wins is the live projection,
    // and WAR comes from futures_preseason.json rather than players.json, which
    // follows the in-season WAR build.
    const sim = preseasonSim();
    const frozenWar = (futuresPreseason && futuresPreseason.heisman_war) || {};
    const pos = { QB: .55, WR: .20, RB: .12, TE: -.08 };
    const scored = rows.map(row => {
      const roster = (players[row.team] && players[row.team].players) || [];
      const player = roster.find(p => p.n === row.player);
      const key = `${row.team}|${row.player}`;
      const war = key in frozenWar ? frozenWar[key] : (player ? player.raw || 0 : 0);
      const po = sim[row.team] || {};
      const score = 1.65 * war + .75 * ((po.avg_wins || 6) / 12) +
        .55 * (po.playoff || 0) + .35 * (po.champ || 0) + (pos[row.position] || 0);
      return { ...row, war, score };
    });
    const lo = Math.min(...scored.map(r => r.score)), hi = Math.max(...scored.map(r => r.score));
    return scored.map(r => ({ ...r, index: hi === lo ? 50 : 20 + 80 * (r.score - lo) / (hi - lo) }));
  }
  const betFlag = `<span class="bet-flag">BET</span>`;
  function renderFutures() {
    setFutureBooks();
    const book = document.getElementById("future-book").value;
    const rows = ((((odds || {}).markets || {})[futureMarket] || {})[book] || []).slice();
    // preseasonSim, NOT simRow: simRow follows DATA.playoff, which the Playoff tab's
    // version toggle rewrites. See the note above regularWinDist.
    const sim = preseasonSim();
    const src = marketSource(futureMarket, book);
    let body = "", note = "";
    if (futureMarket === "heisman") {
      const ranked = heismanIndex(rows).sort((a, b) => b.index - a.index);
      body = ranked.map((r, i) => `<div class="market-row">
        <span class="market-rank">${i + 1}</span><div class="market-team"><b>${esc(r.player)}</b><small><button class="team-link" data-team="${esc(r.team)}">${esc(r.team)}</button> · ${r.position} · ${r.war.toFixed(2)} WAR</small></div>
        <div><small>Heisman index</small><b>${r.index.toFixed(0)}</b></div><div><small>${esc(book)}</small><b>${americanOdds(r.odds)}</b></div>
      </div>`).join("");
      note = "Experimental Heisman index: projected player WAR plus team wins, CFP/title equity and a position effect. It ranks this quoted field; it is not a calibrated award probability.";
    } else if (futureMarket === "win_totals") {
      const priced = rows.map(r => {
        const d = regularWinDist(r.team), overP = d.reduce((s, p, w) => s + (w > r.line ? p : 0), 0);
        const expWins = d.reduce((s, p, w) => s + w * p, 0);
        const io = implied(r.over), iu = implied(r.under), marketOver = io / (io + iu);
        const overEdge = overP - marketOver;
        return { ...r, overP, marketOver, expWins, side: overEdge >= 0 ? "Over" : "Under", edge: Math.abs(overEdge) };
      }).sort((a, b) => b.edge - a.edge);
      const W = BET_RULES.win_total;
      body = priced.map((r, i) => {
        const sideP = r.side === "Over" ? r.overP : 1 - r.overP;
        const bet = sideP > W.minModelP && r.edge >= W.minGap &&
          Math.abs(r.expWins - r.line) >= W.minWinGap;
        return `<div class="market-row">
        <span class="market-rank">${i + 1}</span><div class="market-team">${teamMini(r.team)}${bet ? betFlag : ""}<small>${r.side} ${r.line} · ${americanOdds(r.side === "Over" ? r.over : r.under)}</small></div>
        <div><small>Model</small><b>${pct(sideP, 0)}</b></div><div class="edge"><small>vs no-vig price</small><b>+${pct(r.edge, 1)}</b></div>
      </div>`;
      }).join("");
      note = `Preseason numbers throughout: the win distribution is rebuilt game by game from the week-0 ratings, so conference championships never leak into a sportsbook win-total comparison and nothing here moves once the season starts. A flag also needs the model's expected wins to clear the line by ${BET_RULES.win_total.minWinGap}.`;
    } else {
      const key = futureMarket === "make_cfp" ? "playoff"
        : futureMarket === "conference_title" ? "conf_champ" : "champ";
      const priced = rows.map(r => ({ ...r, modelP: (sim[r.team] || {})[key] || 0,
        marketP: implied(r.odds) })).map(r => ({ ...r, edge: r.modelP - r.marketP }))
        .sort((a, b) => b.edge - a.edge);
      const F = BET_RULES.futures, longOdds = futureMarket === "national_title";
      body = priced.map((r, i) => {
        const bet = !longOdds && r.modelP > F.minModelP && r.edge >= F.minGap;
        return `<div class="market-row">
        <span class="market-rank">${i + 1}</span><div class="market-team">${teamMini(r.team)}${bet ? betFlag : ""}<small>${esc(book)} ${americanOdds(r.odds)}</small></div>
        <div><small>Model</small><b>${pct(r.modelP, 1)}</b></div><div class="edge ${r.edge < 0 ? "negative" : ""}"><small>Model gap</small><b>${r.edge >= 0 ? "+" : ""}${pct(r.edge, 1)}</b></div>
      </div>`;
      }).join("");
      note = "Price gap compares the preseason model with raw implied probability; incomplete futures boards are not de-vigged. Both sides are dated to before week 1 and neither is updated as the season is played.";
    }
    document.getElementById("future-spotlight").innerHTML = `<article class="market-panel"><div class="market-panel-head"><div><span class="eyebrow">Model vs market</span><h3>${futureMarket.replaceAll("_", " ")}</h3></div>${src ? `<a href="${src.url}" target="_blank" rel="noopener">${esc(book)} · ${src.as_of}</a>` : ""}</div><div class="market-list">${body || `<p class="sub">No quoted market is available.</p>`}</div><div class="market-note">${note}</div></article>`;
    wireTeamLinks();
  }
  /* teamMini() stays in app.js */
  function renderOutcomeBands() {
    const all = liveRatings(), dist = cur().playoff.win_dist || {};
    const rows = all.map(r => {
      const d = dist[r.team] || [];
      const floor = d.length ? quantileFromCounts(d, .10) : 0;
      const ceil = d.length ? quantileFromCounts(d, .90) : 0;
      return { ...r, floor, ceil, width: ceil - floor };
    }).sort((a, b) => b.width - a.width || b.power - a.power).slice(0, 18);
    const max = Math.max(...rows.map(r => r.ceil), 12);
    document.getElementById("outcome-bands").innerHTML = `<div class="range-grid">${rows.map(r => `<button class="range-card team-link" data-team="${esc(r.team)}">
      <span class="range-team"><img src="${logoURL(r.team)}" alt="">${esc(r.team)}</span><span class="range-values"><b>${r.floor}</b><i>${r.avg_wins.toFixed(1)} mean</i><b>${r.ceil}</b></span>
      <span class="range-track"><i style="left:${100 * r.floor / max}%;width:${100 * r.width / max}%;background:${color(r.team)}"></i></span><small>${r.width}-win central range</small>
    </button>`).join("")}</div>`;
    wireTeamLinks();
  }


/* ---- Event wiring. It sat just before the #leader-kind listeners. ---- */
  document.querySelectorAll("#future-market .seg-btn").forEach(b => b.addEventListener("click", () => {
    futureMarket = b.dataset.market;
    document.querySelectorAll("#future-market .seg-btn").forEach(x => x.classList.toggle("active", x === b));
    renderFutures();
  }));
  document.getElementById("future-book").addEventListener("change", renderFutures);

/* ---- In render(view), the second branch, after `tracking`. ---- */
    else if (view === "dash") renderDash();

/* ---- In renderAll(), the original three lines (fillConfSelect, renderDash, renderFutures and renderOutcomeBands are the removed calls). ---- */
    fillConfSelect(); fillPlayerSelects(); fillRatingSelects();
    renderDash(); renderPlayoff(); renderScenario(); renderMatchup(); renderTeam();
    renderRatings(); renderPlayers(); renderFutures(); renderOutcomeBands();

