# Conditional CFP rates from the SAME season draws used for the headline odds.
# Records include REG games only; conference/postseason results never enter a bucket.
playoff_record_odds <- function(standings, games, scheduled, fbs, min_samples = 30L) {
  sim_ids <- sort(unique(standings$sim))
  n <- length(sim_ids)
  pending <- scheduled[is.na(scheduled$result), , drop = FALSE]
  last_played <- max(c(0L, scheduled$week[!is.na(scheduled$result)]))
  weeks <- sort(unique(pending$week[pending$week >= last_played]))
  last_week <- max(scheduled$week)
  weeks <- weeks[weeks < last_week]
  checkpoints <- c(list(list(id = "final", label = "End of regular season", week = last_week)),
                   lapply(weeks, function(w) list(id = paste0("week-", w), label = paste("After Week", w), week = w)))
  reg <- games[games$game_type == "REG", , drop = FALSE]
  if (anyNA(reg$result) || any(reg$result == 0)) stop("record odds require completed, untied regular-season draws")
  rows <- setNames(lapply(fbs, function(team) {
    st <- standings[standings$team == team, , drop = FALSE]
    if (nrow(st) != n || anyDuplicated(st$sim)) stop("expected one playoff outcome per team and simulation")
    made <- !is.na(st$seed[match(sim_ids, st$sim)])
    own <- scheduled[scheduled$home_team == team | scheduled$away_team == team, , drop = FALSE]
    played <- own[!is.na(own$result), , drop = FALSE]
    current_w <- sum(ifelse(played$home_team == team, played$result > 0, played$result < 0))
    current_l <- nrow(played) - current_w
    draws <- reg[reg$home_team == team | reg$away_team == team, , drop = FALSE]
    won <- ifelse(draws$home_team == team, draws$result > 0, draws$result < 0)
    buckets <- setNames(lapply(checkpoints, function(cp) {
      included <- draws$week <= cp$week
      counts <- integer(n)
      if (any(included)) {
        totals <- rowsum(as.integer(won[included]), draws$sim[included], reorder = FALSE)
        counts[match(as.integer(rownames(totals)), sim_ids)] <- totals[, 1]
      }
      target <- own[own$week <= cp$week, , drop = FALSE]
      fixed <- target[!is.na(target$result), , drop = FALSE]
      fixed_w <- sum(ifelse(fixed$home_team == team, fixed$result > 0, fixed$result < 0))
      left <- sum(is.na(target$result))
      cells <- lapply(seq.int(fixed_w + left, fixed_w), function(w) {
        at <- counts == w
        samples <- sum(at)
        selected <- sum(made[at])
        list(wins = as.integer(w), losses = as.integer(nrow(target) - w),
             samples = samples, made = selected,
             playoff = if (samples) selected / samples else NULL,
             record_probability = samples / n)
      })
      # All draws must land in exactly one reachable record, and their weighted
      # conditional rate must recover the unrounded headline playoff probability.
      stopifnot(sum(vapply(cells, `[[`, numeric(1), "samples")) == n,
                sum(vapply(cells, `[[`, numeric(1), "made")) == sum(made))
      cells
    }), vapply(checkpoints, `[[`, character(1), "id"))
    list(current_wins = current_w, current_losses = current_l,
         remaining = sum(is.na(own$result)), checkpoints = buckets)
  }), fbs)
  list(generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
       min_samples = min_samples, checkpoints = checkpoints, teams = rows)
}
