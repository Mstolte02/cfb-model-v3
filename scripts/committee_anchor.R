# Preserve the published Top 25 as the baseline, then carry each team's current
# committee-vs-proxy residual into future simulated records. Matchup probabilities
# remain entirely in the football model. No committee information enters preseason.
committee_anchor <- function(cm, schedule, poll, season, record_weight) {
  if (is.null(poll) || !identical(as.integer(poll$season), as.integer(season))) return(NULL)
  released <- as.Date(poll$released)
  cutoff <- as.Date(poll$games_through)
  if (is.na(released) || is.na(cutoff) || released > Sys.Date() || cutoff > released) stop("Invalid committee dates")
  rankings <- poll$rankings
  teams <- vapply(rankings, `[[`, character(1), "team")
  ranks <- vapply(rankings, function(r) as.integer(r$rank), integer(1))
  if (length(teams) != 25 || anyDuplicated(teams) || !setequal(ranks, 1:25) || !all(teams %in% cm$team)) stop("Invalid committee Top 25")
  wins <- games <- setNames(rep(0, nrow(cm)), cm$team)
  for (g in schedule) {
    if (!isTRUE(as.logical(g$f)) || is.null(g$hp) || is.null(g$ap) || is.null(g$d) ||
        as.Date(substr(g$d, 1, 10)) > cutoff || identical(g$st, "postseason")) next
    for (t in intersect(c(g$h, g$a), cm$team)) games[t] <- games[t] + 1
    if (g$hp == g$ap) {
      for (t in intersect(c(g$h, g$a), cm$team)) wins[t] <- wins[t] + .5
    } else {
      winner <- if (g$hp > g$ap) g$h else g$a
      if (winner %in% cm$team) wins[winner] <- wins[winner] + 1
    }
  }
  pct <- ifelse(games > 0, wins / pmax(games, 1), .5)
  proxy <- setNames(record_weight * pct[cm$team] + cm$static, cm$team)
  # Infer rank spacing from today's proxy score-vs-rank relationship, rather than
  # introducing a hand-tuned rank bonus. Keep it strictly positive for exact order.
  slope <- unname(coef(lm(as.numeric(proxy[teams]) ~ ranks))[[2]])
  spacing <- max(abs(slope), .Machine$double.eps^.5)
  intercept <- mean(proxy[teams] + spacing * ranks)
  anchor <- proxy
  anchor[teams] <- intercept - spacing * ranks
  unranked <- setdiff(cm$team, teams)
  anchor[unranked] <- pmin(proxy[unranked], intercept - spacing * 26)
  list(offset = anchor - proxy, baseline = anchor, win_pct = pct,
       rank = setNames(ranks, teams), released = poll$released, games_through = poll$games_through,
       source_url = poll$source_url, rank_spacing = spacing,
       method = "Current CFP Top 25 order, with fixed committee residuals carried into simulated final records")
}
