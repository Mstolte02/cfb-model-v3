# Resolve championship participants with cfbseedR's conference-specific cascades,
# then PLAY the championship. First place in REG standings is not a title win.
conference_teams <- function(teams) {
  east <- c("App State", "Coastal Carolina", "Georgia Southern", "Georgia State",
            "James Madison", "Marshall", "Old Dominion")
  teams$conf_division <- ifelse(teams$conference == "Sun Belt",
                               ifelse(teams$team %in% east, "East", "West"), NA_character_)
  teams
}

conference_standings <- function(games, teams, scheduled, analytics) {
  # cfbseedR otherwise infers conference games from conference membership. Respect
  # explicit non-conference designations, notably Army-Navy, for the race only.
  conf <- setNames(teams$conference, teams$team)
  same_league <- !is.na(conf[scheduled$home_team]) & !is.na(conf[scheduled$away_team]) &
    conf[scheduled$home_team] == conf[scheduled$away_team]
  nonconf <- scheduled[scheduled$game_type == "REG" & !scheduled$conference_game & same_league, ]
  pair <- function(g) paste(g$home_team, g$away_team, g$week)
  race_games <- games[!(games$game_type == "REG" & pair(games) %in% pair(nonconf)), ]
  ranked <- cfbseedR::cfb_standings(race_games, teams, tiebreaker_depth = "POINTS",
    tiebreaker_data = list(analytics_ratings = analytics), verbosity = "NONE")
  overall <- cfbseedR::cfb_standings(games, teams, ranks = "NONE")
  ix <- match(paste(ranked$sim, ranked$team), paste(overall$sim, overall$team))
  # Non-conference games still count toward committee records.
  for (key in c("games", "wins", "losses", "ties", "win_pct", "pd")) ranked[[key]] <- overall[[key]][ix]
  ranked
}

simulate_conference_titles <- function(games, teams, scheduled, analytics, results, spec) {
  teams <- conference_teams(teams)
  standings <- conference_standings(games, teams, scheduled, analytics)
  candidates <- standings[standings$conf_rank %in% 1:2 & !is.na(standings$conf_rank), ]
  pools <- split(candidates, paste(candidates$sim, candidates$conference))
  title_week <- max(scheduled$week) + 1L
  campus <- c("American Athletic", "Conference USA", "Mountain West", "Sun Belt")
  titles <- lapply(pools, function(s) {
    if (nrow(s) != 2L) stop("A conference championship requires two participants")
    s <- s[order(s$conf_rank), ]
    data.frame(sim = s$sim[1], game_type = "CONF_CHAMP", week = title_week,
      home_team = s$team[1], away_team = s$team[2], result = NA_real_,
      home_points = NA_real_, away_points = NA_real_,
      neutral = as.integer(!s$conference[1] %in% campus), stringsAsFactors = FALSE)
  })
  titles <- do.call(rbind, titles)
  if (is.null(titles)) stop("No conference championship participants")
  titles <- results(teams, titles, title_week, spec)$games
  stopifnot(!anyNA(titles$result), all(titles$result != 0))
  combined <- dplyr::bind_rows(games, titles)
  final <- conference_standings(combined, teams, scheduled, analytics)
  # Every conference has exactly one champion in every draw, decided by its game.
  winners <- final[final$conf_champ %in% TRUE, ]
  stopifnot(nrow(winners) == length(pools),
            !anyDuplicated(paste(winners$sim, winners$conference)))
  list(games = combined, standings = final)
}
