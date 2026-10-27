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

# Pair each game's winner with each team's eventual playoff selection, using the
# identical simulation IDs. These are conditional comparisons, not independent
# reruns with a forced winner. Retain both denominators and null unsampled outcomes.
playoff_game_impacts <- function(standings, games, scheduled, fbs,
                                 metadata = list(), min_samples = 30L, threshold = .01) {
  sim_ids <- sort(unique(standings$sim))
  n <- length(sim_ids)
  selected <- vapply(fbs, function(team) {
    st <- standings[standings$team == team, , drop = FALSE]
    if (nrow(st) != n || anyDuplicated(st$sim)) stop("expected one playoff outcome per team and simulation")
    !is.na(st$seed[match(sim_ids, st$sim)])
  }, logical(n))
  # vapply collapses its result to a vector for a single simulation.
  selected <- matrix(selected, nrow = n, dimnames = list(NULL, fbs))
  baseline <- colMeans(selected)
  eligible <- names(baseline)[baseline >= threshold]
  key <- function(week, home, away) paste(week, home, away, sep = "\r")
  reg <- games[games$game_type == "REG", , drop = FALSE]
  pending <- scheduled[scheduled$game_type == "REG" & is.na(scheduled$result), , drop = FALSE]
  draws <- split(seq_len(nrow(reg)), key(reg$week, reg$home_team, reg$away_team))
  meta <- setNames(metadata, vapply(metadata, function(g) key(g$w, g$h, g$a), character(1)))
  rows <- lapply(seq_len(nrow(pending)), function(i) {
    g <- pending[i, ]
    id <- key(g$week, g$home_team, g$away_team)
    sim_game <- reg[draws[[id]], , drop = FALSE]
    if (nrow(sim_game) != n || anyDuplicated(sim_game$sim) || anyNA(sim_game$result) || any(sim_game$result == 0)) {
      stop("expected one untied game result per simulation: ", id)
    }
    home_won <- sim_game$result[match(sim_ids, sim_game$sim)] > 0
    nh <- sum(home_won); na <- n - nh
    ph <- if (nh) colMeans(selected[home_won, , drop = FALSE]) else rep(NA_real_, length(fbs))
    pa <- if (na) colMeans(selected[!home_won, , drop = FALSE]) else rep(NA_real_, length(fbs))
    names(ph) <- names(pa) <- fbs
    delta <- ph - pa
    null_if_na <- function(x) if (is.finite(x)) unname(x) else NULL
    keep <- union(eligible, intersect(c(g$home_team, g$away_team), fbs))
    effects <- setNames(lapply(keep, function(team) {
      se <- if (nh && na) sqrt(ph[team] * (1 - ph[team]) / nh + pa[team] * (1 - pa[team]) / na) else NA_real_
      list(home_win = null_if_na(ph[team]), away_win = null_if_na(pa[team]),
           swing = null_if_na(delta[team]), standard_error = null_if_na(se))
    }), keep)
    participants <- intersect(c(g$home_team, g$away_team), fbs)
    info <- meta[[id]]
    list(id = if (!is.null(info$id)) as.character(info$id) else paste(g$week, g$home_team, g$away_team, sep = "|"),
         week = as.integer(g$week), date = info$d,
         home = g$home_team, away = g$away_team,
         samples_home = nh, samples_away = na, home_win_probability = nh / n,
         supported = nh >= min_samples && na >= min_samples,
         combined_swing = if (nh && na) sum(abs(delta[participants])) else NULL,
         effects = effects)
  })
  list(min_samples = min_samples, rooting_threshold = threshold,
       # Simultaneous 95% normal intervals across every tested game for a team.
       # A per-game 1.96 cutoff produces false rooting interests when scanning
       # hundreds of otherwise unrelated outcomes.
       rooting_min_swing = .02,
       rooting_z = qnorm(1 - .05 / (2 * max(1L, length(rows)))),
       rooting_teams = as.list(eligible),
       baseline = as.list(baseline), games = rows)
}
