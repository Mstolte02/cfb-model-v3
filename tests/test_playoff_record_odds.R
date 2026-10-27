source("scripts/playoff_record_odds.R")
# Four equiprobable seasons: A makes the field in 2, B in 1. A's 2-0
# record is always selected; its 1-1 record is selected once in three draws.
scheduled <- data.frame(home_team = c("A", "A", "B"), away_team = c("B", "C", "C"),
                        week = c(1L, 2L, 3L), result = c(7, NA, NA), game_type = "REG")
games <- scheduled[rep(1:3, 4), ]
games$sim <- rep(1:4, each = 3)
games$result <- c(7, 3, 3, 7, -3, 3, 7, -3, -3, 7, -3, -3)
standings <- expand.grid(sim = 1:4, team = c("A", "B", "C"), stringsAsFactors = FALSE)
standings$seed <- c(1L, 2L, NA, NA, NA, NA, 3L, NA, 4L, 4L, 4L, 4L)
out <- playoff_record_odds(standings, games, scheduled, c("A", "B", "C"))
a <- out$teams$A
stopifnot(a$current_wins == 1, a$current_losses == 0, a$remaining == 1,
          a$checkpoints$final[[1]]$wins == 2, a$checkpoints$final[[1]]$playoff == 1,
          a$checkpoints$final[[2]]$samples == 3, a$checkpoints$final[[2]]$made == 1,
          abs(a$checkpoints$final[[2]]$playoff - 1/3) < 1e-12,
          sum(vapply(a$checkpoints$final, `[[`, numeric(1), "record_probability")) == 1)
# B's winning and losing records use their own conditional denominators.
b <- out$teams$B
stopifnot(b$current_losses == 1, b$checkpoints$final[[1]]$playoff == 0,
          b$checkpoints$final[[2]]$playoff == .5)
# Adding an unobserved reachable record does not invent odds.
extra <- scheduled
extra$result[1] <- NA
unseen <- playoff_record_odds(standings, games, extra, c("A", "B", "C"))
stopifnot(unseen$teams$A$checkpoints$final[[3]]$samples == 0,
          is.null(unseen$teams$A$checkpoints$final[[3]]$playoff))
# Bye week records, current locks, and weekly partitions preserve headline odds.
for (team in names(out$teams)) {
  for (cp in out$teams[[team]]$checkpoints) {
    stopifnot(sum(vapply(cp, `[[`, numeric(1), "samples")) == 4,
              sum(vapply(cp, `[[`, numeric(1), "made")) == sum(!is.na(standings$seed[standings$team == team])))
  }
}
# Ignore title/playoff games when bucketing the REG record.
post <- games[1:4, ]; post$game_type <- "CONF_CHAMP"; post$week <- 4L
same <- playoff_record_odds(standings, rbind(games, post), scheduled, c("A", "B", "C"))
stopifnot(identical(out$teams, same$teams))
# Ordering of simulation rows cannot change conditional rates.
reverse <- playoff_record_odds(standings[nrow(standings):1, ], games[nrow(games):1, ], scheduled, c("A", "B", "C"))
stopifnot(identical(out$teams, reverse$teams))
# Once every result is fixed, one record remains and recovers headline odds.
fixed <- scheduled; fixed$result <- c(7, -3, -3)
fixed_games <- fixed[rep(1:3, 4), ]; fixed_games$sim <- rep(1:4, each = 3)
finished <- playoff_record_odds(standings, fixed_games, fixed, c("A", "B", "C"))
stopifnot(length(finished$checkpoints) == 1, length(finished$teams$A$checkpoints$final) == 1,
          finished$teams$A$checkpoints$final[[1]]$playoff == .5)
cat("Playoff record odds: conditional denominators, locks, zero samples, checkpoints, postseason exclusion, draw alignment passed\n")
