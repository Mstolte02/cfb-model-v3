/* Pure selectors shared by the playoff explorer and its checks. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PlayoffPaths = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const finite = x => typeof x === "number" && Number.isFinite(x);
  function percent(p) {
    if (!finite(p)) return "—";
    if (p === 0) return "0%";
    if (p === 1) return "100%";
    if (p < .001) return "<0.1%";
    if (p > .999) return ">99.9%";
    return `${(100 * p).toFixed(1)}%`;
  }
  function modalRecords(cells) {
    const max = Math.max(0, ...cells.map(c => c.samples));
    return max ? cells.filter(c => c.samples === max).sort((a, b) => a.losses - b.losses) : [];
  }
  function lossColumns(rows) {
    const losses = rows.flatMap(cells => cells.map(c => c.losses));
    if (!losses.length) return [];
    return Array.from({length: Math.max(...losses) + 1}, (_, i) => i);
  }
  function rankGames(games) {
    return [...games].sort((a, b) => Number(b.supported) - Number(a.supported) ||
      (finite(b.combined_swing) ? b.combined_swing : -1) - (finite(a.combined_swing) ? a.combined_swing : -1) ||
      a.home.localeCompare(b.home) || a.away.localeCompare(b.away));
  }
  function rootingGuide(impacts, team, week) {
    if (!(impacts.rooting_teams || []).includes(team)) return [];
    return impacts.games.filter(g => (week === "all" || g.week === Number(week)) &&
      g.home !== team && g.away !== team && g.supported && g.effects[team] &&
      // Keep the guide focused on playoff contenders. Associations with games
      // between two teams below the eligibility threshold are not recommendations.
      (impacts.rooting_teams.includes(g.home) || impacts.rooting_teams.includes(g.away)))
      .map(g => {
        const e = g.effects[team];
        if (!finite(e.swing) || !finite(e.home_win) || !finite(e.away_win)) return null;
        const swing = Math.abs(e.swing);
        // Exclude negligible differences and differences smaller than the Monte
        // Carlo sampling uncertainty, accounting for the full set of games scanned.
        // Do not manufacture a rooting interest from repeated comparisons.
        if (swing < impacts.rooting_min_swing || !finite(e.standard_error) || swing <= (impacts.rooting_z || 3.8) * e.standard_error) return null;
        return {game: g, team, winner: e.swing > 0 ? g.home : g.away,
          other: e.swing > 0 ? g.away : g.home,
          preferred: e.swing > 0 ? e.home_win : e.away_win,
          opposite: e.swing > 0 ? e.away_win : e.home_win, swing};
      }).filter(Boolean).sort((a, b) => b.swing - a.swing || a.game.week - b.game.week || a.winner.localeCompare(b.winner));
  }
  return {percent, modalRecords, lossColumns, rankGames, rootingGuide};
});
