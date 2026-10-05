/* Team Overview helpers: pure functions, shared by app.js and tests/test_team_overview.cjs.
   Nothing here draws; app.js turns these numbers into the cards. */
(function (root) {
  const INJURY = new Set(["out", "doubtful", "questionable"]);

  /* Empirical percentile rank of x among sorted values: share at or below, ties at
     their midrank. No distributional assumption. */
  function percentile(sorted, x) {
    if (!sorted || !sorted.length || !Number.isFinite(x)) return null;
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] < x) lo = m + 1; else hi = m; }
    const below = lo;
    hi = sorted.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] <= x) lo = m + 1; else hi = m; }
    return 100 * (below + 0.5 * (lo - below)) / sorted.length;
  }

  /* Index of the fixed bin holding x; the last bin is closed on the right. */
  function binIndex(bins, x) {
    if (!bins || !bins.length || !Number.isFinite(x)) return -1;
    if (x < bins[0].lo) return 0;
    for (let i = 0; i < bins.length; i++) if (x < bins[i].hi) return i;
    return bins.length - 1;
  }

  /* Nearest historical team-weeks to x, one per team-season (its closest week), from
     the export's [season, week, team, power] rows. */
  function comparables(observations, x, n = 5, exclude = null) {
    const best = new Map();
    for (const [season, week, team, power] of observations || []) {
      if (exclude && exclude.season === season && exclude.team === team) continue;
      const key = season + "\u0000" + team, d = Math.abs(power - x);
      const cur = best.get(key);
      if (!cur || d < cur.d) best.set(key, {season, week, team, power, d});
    }
    return [...best.values()].sort((a, b) => a.d - b.d).slice(0, n);
  }

  const injured = p => !!p.out || INJURY.has(p.inj);

  /* Trending up / down on measured WAR change. Injured players are excluded from both:
     an availability cut is not a performance decline, and they belong in the injury
     report instead. */
  function movers(players, n = 5, floor = 0.0005) {
    const healthy = (players || []).filter(p => !injured(p) && Number.isFinite(p.dwin));
    return {
      up: healthy.filter(p => p.dwin >= floor).sort((a, b) => b.dwin - a.dwin).slice(0, n),
      down: healthy.filter(p => p.dwin <= -floor).sort((a, b) => a.dwin - b.dwin).slice(0, n),
    };
  }

  /* Players on an injury report, with WAR when he plays, WAR counted now, and the gap. */
  function injuryReport(players) {
    const order = {out: 0, doubtful: 1, questionable: 2};
    return (players || []).filter(injured).map(p => {
      const status = p.out ? "out" : p.inj;
      const counted = Number.isFinite(p.win) ? p.win : (p.w || 0);
      const base = Number.isFinite(p.base) ? p.base : null;
      return {...p, status, counted, base, impact: base == null ? null : counted - base};
    }).sort((a, b) => (order[a.status] - order[b.status]) ||
                      ((a.impact ?? 0) - (b.impact ?? 0)));
  }

  /* Every unit the roster has, ranked against the league table (GROUP_LEAGUE). */
  function unitRankings(byGroup, league, team, change = {}) {
    return Object.entries(byGroup || {}).flatMap(([g, v]) => {
      const L = league && league[g];
      if (!L || L.rank[team] == null) return [];
      const rank = L.rank[team];
      const pct = L.n > 1 ? 100 * (L.n - rank) / (L.n - 1) : 100;
      const tier = pct >= 85 ? "elite" : pct >= 60 ? "above" : pct >= 40 ? "average" : "below";
      return [{group: g, war: v, rank, n: L.n, pct, tier, change: change[g] ?? null}];
    }).sort((a, b) => a.rank - b.rank || a.group.localeCompare(b.group));
  }

  /* Completed games in order, each with its pregame and postgame Power Rating where
     the weekly history has them (pre = the last snapshot before that week). */
  function journey(card) {
    const pts = (card.points || []).slice().sort((a, b) => a.week - b.week);
    return (card.games || []).slice().sort((a, b) => a.week - b.week).map(g => {
      const before = pts.filter(p => p.week < g.week).pop() || null;
      const after = pts.find(p => p.week === g.week) || null;
      return {...g, pre: before ? before.power : null, post: after ? after.power : null};
    });
  }

  const api = {percentile, binIndex, comparables, movers, injuryReport, unitRankings, journey};
  root.TeamOverview = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
