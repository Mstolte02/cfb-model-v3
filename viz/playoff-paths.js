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
  return {percent, modalRecords, lossColumns, rankGames};
});
