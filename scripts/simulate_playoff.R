#!/usr/bin/env Rscript

# Build both public CFP projections with cfbseedR's full-season simulator.
#
# Both runs are the same engine and the same selection rule. The preseason run
# simulates every game from the v5 ensemble's week-0 state (initial ratings, no form,
# empty PFF and WAR tables) and ranks teams by the week-0 Power Rating. The current
# run locks completed scores, uses the live ensemble state, and ranks teams by the
# current Power Rating. Matchup probabilities go through cfbseedR's compute_results.
#
# Run: Rscript scripts/simulate_playoff.R [simulations] [both|preseason|current]

suppressPackageStartupMessages({
  library(cfbseedR)
  library(jsonlite)
})
if (utils::packageVersion("cfbseedR") < "0.2.0") {
  stop("cfbseedR >= 0.2.0 is required for the 2026 autobid policy")
}

args <- commandArgs(trailingOnly = TRUE)
n_sims <- if (length(args) && grepl("^[0-9]+$", args[[1]])) as.integer(args[[1]]) else 2000L
mode_arg <- if (length(args) >= 2L) args[[2]] else "both"
if (!mode_arg %in% c("both", "preseason", "current")) {
  stop("mode must be one of: both, preseason, current")
}
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])), ".."))
viz <- file.path(root, "viz", "data")
source(file.path(root, "scripts", "playoff_record_odds.R"))

# cfbseedR splits work into chunks but deliberately leaves execution sequential
# unless the caller selects a future plan. Use the runner's available cores while
# capping workers to avoid exhausting memory on the full FBS schedule.
workers <- max(1L, min(4L, as.integer(future::availableCores())))
if (workers > 1L) future::plan(future::multisession, workers = workers)
message("Simulation workers: ", workers, "; runs per projection: ", n_sims)

read_json <- function(path) jsonlite::fromJSON(path, simplifyVector = FALSE)
model <- read_json(file.path(viz, "model_v4.json"))
team_meta <- read_json(file.path(viz, "teams.json"))
schedule <- read_json(file.path(viz, "schedule.json"))

fbs <- intersect(names(model$teams), names(team_meta))
vectors <- do.call(rbind, lapply(fbs, function(team) unlist(model$teams[[team]], use.names = FALSE)))
rownames(vectors) <- fbs
logit_coef <- unlist(model$logistic$coef, use.names = FALSE)
margin_coef <- unlist(model$margin$coef, use.names = FALSE)
static_logit <- drop(vectors %*% logit_coef) + model$logistic$intercept
static_margin <- drop(vectors %*% margin_coef) + model$margin$intercept
names(static_logit) <- names(static_margin) <- fbs

legacy_probability <- function(home, away, neutral, spec) {
  hfa_logit <- ifelse(neutral, 0, model$logistic$hfa)
  hfa_margin <- ifelse(neutral, 0, model$margin$hfa)
  z <- spec$logit[home] - spec$logit[away] + hfa_logit
  margin <- spec$margin[home] - spec$margin[away] + hfa_margin
  p <- model$ens_w * plogis(z) + (1 - model$ens_w) * pnorm(margin / model$margin$sigma)
  p <- pmin(1 - 1e-8, pmax(1e-8, p))
  p <- plogis((model$probability_scale %||% 1) * qlogis(p))
  if (!is.null(spec$dynamic)) {
    pd <- plogis(spec$dynamic[home] - spec$dynamic[away] + hfa_logit)
    p <- (1 - spec$blend) * p + spec$blend * pd
  }
  unname(pmin(1 - 1e-6, pmax(1e-6, p)))
}

`%||%` <- function(x, y) if (is.null(x)) y else x

# The v5 live ensemble, a port of scripts/ensemble_replay.probability: each member's
# logistic stack over [prior_level, elo_change, dO, dD, hfa], averaged unweighted.
# Games with a team outside the ensemble (FCS opponents) keep the v4 formula.
ensemble_tables <- function(ensemble) {
  # v5.1: a PFF table in the state switches every member to its PFF stack.
  pff <- ensemble$state$pff
  # v5.2: in-season player WAR; with a PFF table it switches members to stack_war.
  war <- ensemble$state$war
  lapply(ensemble$members, function(m) {
    init <- unlist(m$initial)
    r <- unlist(ensemble$state$ratings[[m$name]])[names(init)]
    form <- ensemble$state$form[[m$name]]
    O <- D <- n <- setNames(rep(NA_real_, length(init)), names(init))
    for (t in intersect(names(form), names(init))) {
      O[t] <- form[[t]][[1]]
      D[t] <- form[[t]][[2]]
      n[t] <- form[[t]][[3]]
    }
    Op <- Dp <- setNames(rep(0, length(init)), names(init))
    for (t in intersect(names(pff), names(init))) {
      Op[t] <- pff[[t]][[1]]
      Dp[t] <- pff[[t]][[2]]
    }
    Wv <- setNames(rep(0, length(init)), names(init))
    for (t in intersect(names(war), names(init))) Wv[t] <- war[[t]]
    # Slates before stack_war_from_slate keep the legacy WAR stack they were graded
    # on, as in scripts/ensemble_replay._stack.
    slate <- ensemble$state$slate
    war_stack <- if (!is.null(m$stack_war_legacy) && !is.null(m$stack_war_from_slate) &&
                     !is.null(slate) && slate < m$stack_war_from_slate) m$stack_war_legacy else m$stack_war
    st <- if (!is.null(pff) && !is.null(war) && !is.null(war_stack)) war_stack
          else if (!is.null(pff) && !is.null(m$stack_pff)) m$stack_pff else m
    list(init = init, r = r, O = O, D = D, n = n, Op = Op, Dp = Dp, Wv = Wv,
         columns = unlist(st$columns), scale = unlist(st$scale), coef = unlist(st$coef))
  })
}

probability <- function(home, away, neutral, spec) {
  p <- legacy_probability(home, away, neutral, spec)
  if (is.null(spec$ensemble)) return(p)
  known <- names(spec$ensemble[[1]]$init)
  rated <- home %in% known & away %in% known
  if (!any(rated)) return(p)
  h <- home[rated]
  a <- away[rated]
  hfa <- ifelse(neutral[rated], 0, 1)
  total <- 0
  for (m in spec$ensemble) {
    have <- !is.na(m$n[h]) & !is.na(m$n[a]) &
      m$n[h] >= spec$min_form & m$n[a] >= spec$min_form
    x <- list(prior_level = m$init[h] - m$init[a],
              elo_change = (m$r[h] - m$init[h]) - (m$r[a] - m$init[a]),
              dO = ifelse(have, m$O[h] - m$O[a], 0),
              dD = ifelse(have, m$D[h] - m$D[a], 0),
              hfa = hfa,
              pff_O_diff = m$Op[h] - m$Op[a],
              pff_D_diff = m$Dp[h] - m$Dp[a],
              war_delta_diff = m$Wv[h] - m$Wv[a])
    z <- 0
    for (k in seq_along(m$columns)) z <- z + m$coef[k] * x[[m$columns[k]]] / m$scale[k]
    total <- total + plogis(z)
  }
  p[rated] <- unname(pmin(1 - 1e-6, pmax(1e-6, total / length(spec$ensemble))))
  p
}
round_away <- function(x) as.integer(ifelse(x < 0, floor(x), ceiling(x)))

model_results <- function(teams, games, week_num, spec) {
  fill <- games$week == week_num & is.na(games$result)
  if (!any(fill)) return(list(teams = teams, games = games))
  home <- games$home_team[fill]
  away <- games$away_team[fill]
  neutral <- !is.na(games$neutral[fill]) & games$neutral[fill] == 1
  p <- probability(home, away, neutral, spec)
  sigma <- spec$sigma %||% model$margin$sigma
  mean_margin <- qnorm(p) * sigma
  result <- round_away(rnorm(length(p), mean_margin, sigma))
  post <- games$game_type[fill] != "REG"
  tied <- post & result == 0L
  if (any(tied)) result[tied] <- ifelse(runif(sum(tied)) < p[tied], 3L, -3L)
  games$result[fill] <- result
  list(teams = teams, games = games)
}

make_inputs <- function(mode) {
  rows <- lapply(schedule, function(g) {
    played <- mode == "current" && isTRUE(as.logical(g$f)) && !is.null(g$hp) && !is.null(g$ap)
    data.frame(
      season = 2026L,
      game_type = "REG",
      week = as.integer(g$w),
      home_team = g$h,
      away_team = g$a,
      result = if (played) as.numeric(g$hp) - as.numeric(g$ap) else NA_real_,
      home_points = if (played) as.numeric(g$hp) else NA_real_,
      away_points = if (played) as.numeric(g$ap) else NA_real_,
      neutral = as.integer(isTRUE(as.logical(g$n))),
      stringsAsFactors = FALSE
    )
  })
  games <- do.call(rbind, rows)
  opponents <- unique(c(games$home_team, games$away_team))
  all_teams <- union(fbs, opponents)
  conference <- vapply(all_teams, function(team) {
    if (team %in% fbs) team_meta[[team]]$conference %||% "FBS Independents" else "FBS Independents"
  }, character(1))
  teams <- data.frame(
    team = all_teams,
    conference = conference,
    division = ifelse(all_teams %in% fbs, "FBS", "FCS"),
    postseason_eligible = all_teams %in% fbs,
    stringsAsFactors = FALSE
  )

  # Unlisted opponents still need a probability while their games are simulated.
  fallback_logit <- min(static_logit) - 3
  fallback_margin <- min(static_margin) - 20
  logit <- setNames(rep(fallback_logit, length(all_teams)), all_teams)
  margin <- setNames(rep(fallback_margin, length(all_teams)), all_teams)
  logit[fbs] <- static_logit
  margin[fbs] <- static_margin
  spec <- list(logit = logit, margin = margin, dynamic = NULL, blend = 0)
  if (mode == "current" && !is.null(model$dynamic$ratings)) {
    dynamic <- setNames(rep(fallback_logit, length(all_teams)), all_teams)
    current <- unlist(model$dynamic$ratings, use.names = TRUE)
    dynamic[names(current)] <- current
    spec$dynamic <- dynamic
    spec$blend <- model$dynamic$blend %||% 0
  }
  if (!is.null(model$ensemble$state)) {
    ensemble <- model$ensemble
    # The week-0 state, as capture_market_snapshot draws the Preseason Power Ratings:
    # initial ratings, no form, empty PFF and WAR tables, slate 1.
    if (mode == "preseason") {
      ensemble$state <- list(
        ratings = setNames(lapply(ensemble$members, `[[`, "initial"),
                           vapply(ensemble$members, `[[`, character(1), "name")),
        form = list(), pff = setNames(list(), character()), war = setNames(list(), character()),
        slate = 1L)
    }
    spec$ensemble <- ensemble_tables(ensemble)
    spec$min_form <- model$ensemble$min_form_games %||% 2
    spec$sigma <- model$ensemble$margin_sigma
  }
  # Team quality in the committee proxy is the published Power Rating (ratings.json,
  # the mean neutral-site win probability against every rated team): the week-0 row
  # for the preseason run, today's for the current run.
  rating <- {
    ratings <- read_json(file.path(viz, "ratings.json"))
    power <- if (mode == "preseason") {
      week0 <- Filter(function(h) identical(as.integer(h$week), 0L), ratings$history)
      if (!length(week0)) stop("ratings.json has no week-0 (Preseason) power ratings")
      week0[[1]]$teams
    } else ratings$teams
    power <- setNames(vapply(power, function(x) as.numeric(x$power), numeric(1)),
                      vapply(power, function(x) x$team, character(1)))
    missing <- setdiff(fbs, names(power))
    if (length(missing)) stop("ratings.json has no power rating for: ", paste(missing, collapse = ", "))
    power[fbs]
  }
  rankings <- data.frame(team = fbs, rank = rank(-rating, ties.method = "first"))
  analytics <- data.frame(team = fbs, rating = as.numeric(rating))
  list(games = games, teams = teams, rankings = rankings, analytics = analytics, spec = spec)
}

# Committee proxy, the h2h-free fit from scripts/fit_committee.py on every published
# committee ranking 2014-2025 (its provisional_weights).
# Each simulation ranks teams on its own simulated record, so seeding follows how the
# schedule plays out: score = win_pct + quality + schedule strength + power league.
W_WINPCT <- 10.3397
W_RATING <- 0.2262
W_SOS <- 0.7362
W_P4 <- 1.0864
FCS_OPP_RATING <- -2.0
P4_CONFS <- c("ACC", "Big 12", "Big Ten", "SEC")
G6_CONFS <- c("American Athletic", "Conference USA", "Mid-American", "Mountain West", "Pac-12", "Sun Belt")

# The schedule-only terms: quality z, mean opponent quality z (FCS opponents count
# FCS_OPP_RATING), and the power-league flag. Notre Dame carries the power flag, as
# in scripts/fit_committee.POWER_INDEPENDENTS.
static_committee <- function(input) {
  z <- setNames(as.numeric(scale(input$analytics$rating)), input$analytics$team)
  g <- input$games
  opp <- function(team, other) ifelse(other %in% names(z), z[other], FCS_OPP_RATING)
  sos <- vapply(fbs, function(team) {
    vals <- c(opp(team, g$away_team[g$home_team == team]), opp(team, g$home_team[g$away_team == team]))
    if (length(vals)) mean(vals) else 0
  }, numeric(1))
  sos_z <- (sos - mean(sos)) / sd(sos)
  conf <- vapply(fbs, function(team) team_meta[[team]]$conference %||% "FBS Independents", character(1))
  p4 <- as.numeric(conf %in% P4_CONFS | fbs == "Notre Dame")
  data.frame(team = fbs, conference = conf, rating_z = z[fbs], sos_z = sos_z, p4 = p4,
             static = W_RATING * z[fbs] + W_SOS * sos_z + W_P4 * p4, stringsAsFactors = FALSE)
}

# Selection and bracket for every simulation: four P4 champions, the best-scoring G6
# team, then at-large by score; straight seeding; first round at the higher seed,
# quarterfinals on at neutral sites. Returns one row per (sim, team) with seed and exit.
select_and_play <- function(standings, input) {
  cm <- static_committee(input)
  st <- standings[standings$team %in% fbs, c("sim", "team", "win_pct", "conf_champ")]
  st$score <- W_WINPCT * st$win_pct + cm$static[match(st$team, cm$team)]
  st$conference <- cm$conference[match(st$team, cm$team)]
  sims <- split(st, st$sim)
  seeds <- t(vapply(sims, function(s) {
    s <- s[order(-s$score, s$team), ]
    auto <- character()
    for (c in P4_CONFS) {
      champ <- s$team[s$conference == c & s$conf_champ %in% TRUE]
      if (!length(champ)) champ <- s$team[s$conference == c]
      auto <- c(auto, champ[1])
    }
    auto <- c(auto, s$team[s$conference %in% G6_CONFS][1])
    field <- c(auto, head(setdiff(s$team, auto), 12L - length(auto)))
    s$team[s$team %in% field]
  }, character(12)))
  play <- function(a, b, home) {
    p <- probability(a, b, rep(!home, length(a)), input$spec)
    ifelse(runif(length(a)) < p, a, b)
  }
  r1 <- lapply(list(c(8, 9), c(5, 12), c(6, 11), c(7, 10)),
               function(k) play(seeds[, k[1]], seeds[, k[2]], TRUE))
  qf <- list(play(seeds[, 1], r1[[1]], FALSE), play(seeds[, 4], r1[[2]], FALSE),
             play(seeds[, 3], r1[[3]], FALSE), play(seeds[, 2], r1[[4]], FALSE))
  sf <- list(play(qf[[1]], qf[[2]], FALSE), play(qf[[4]], qf[[3]], FALSE))
  champ <- play(sf[[1]], sf[[2]], FALSE)
  sim_ids <- as.integer(names(sims))
  out <- standings[standings$team %in% fbs, ]
  key <- paste(out$sim, out$team)
  at <- function(teams) paste(rep(sim_ids, ncol(teams)), as.vector(teams))
  out$seed <- NA_integer_
  out$seed[match(at(seeds), key)] <- rep(seq_len(12), each = length(sim_ids))
  out$exit <- 0L
  out$exit[!is.na(out$seed)] <- 1L
  out$exit[match(at(cbind(seeds[, 1:4], do.call(cbind, r1))), key)] <- 2L
  out$exit[match(at(do.call(cbind, qf)), key)] <- 3L
  out$exit[match(at(do.call(cbind, sf)), key)] <- 4L
  out$exit[match(at(cbind(champ)), key)] <- 5L
  out
}

safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) mean(x) else 0
}

# Per-simulation seed is NA when a team misses the playoff. That is a FALSE event,
# not missing information. Using mean(st$seed == seed) directly turns every seed
# probability into NA as soon as a team misses once.
event_rate <- function(x) {
  if (!length(x)) return(0)
  mean(replace(x, is.na(x), FALSE))
}

row_number <- function(row, key, fallback = 0) {
  value <- row[[key]]
  if (!length(value) || is.na(value[[1]]) || !is.finite(value[[1]])) fallback else as.numeric(value[[1]])
}

build_bracket <- function(rows) {
  p4 <- c("ACC", "Big 12", "Big Ten", "SEC")
  g6 <- c("American Athletic", "Conference USA", "Mid-American", "Mountain West", "Pac-12", "Sun Belt")
  chosen <- character()
  take <- function(team) {
    if (!is.null(team) && length(team) && !team %in% chosen) chosen <<- c(chosen, team)
  }
  for (conference in p4) {
    pool <- rows[vapply(rows, function(x) identical(x$conference, conference), logical(1))]
    if (length(pool)) {
      score <- vapply(pool, row_number, numeric(1), key = "conf_champ")
      take(pool[[which.max(score)]]$team)
    }
  }
  pool <- rows[vapply(rows, function(x) x$conference %in% g6, logical(1))]
  if (length(pool)) {
    score <- vapply(pool, row_number, numeric(1), key = "playoff")
    take(pool[[which.max(score)]]$team)
  }
  ordered <- rows[order(-vapply(rows, row_number, numeric(1), key = "playoff"),
                        vapply(rows, row_number, numeric(1), key = "power_rank", fallback = Inf))]
  for (row in ordered) if (length(chosen) < 12) take(row$team)
  field <- rows[vapply(rows, function(x) x$team %in% chosen[seq_len(min(12, length(chosen)))], logical(1))]
  if (length(field) < 12) stop("cannot build a 12-team bracket: only ", length(field), " teams selected")
  seeds <- rep(NA_character_, 12)
  remaining <- field
  for (seed in seq_len(12)) {
    if (!length(remaining)) break
    score <- vapply(remaining, function(x) row_number(list(value = x$seeds[[seed]]), "value", -Inf), numeric(1))
    pick <- which.max(score)
    if (!length(pick)) pick <- 1L
    seeds[[seed]] <- remaining[[pick]]$team
    remaining <- remaining[-pick]
  }
  game <- function(round, top = NULL, bottom = NULL, site = NULL) list(round = round, top = top, bottom = bottom, site = site)
  list(
    seeds = unname(seeds),
    games = list(
      game("R1", seeds[[8]], seeds[[9]], seeds[[8]]), game("R1", seeds[[5]], seeds[[12]], seeds[[5]]),
      game("R1", seeds[[6]], seeds[[11]], seeds[[6]]), game("R1", seeds[[7]], seeds[[10]], seeds[[7]]),
      game("QF", seeds[[1]]), game("QF", seeds[[4]]), game("QF", seeds[[3]]), game("QF", seeds[[2]]),
      game("SF"), game("SF"), game("F")
    ),
    feeds = list(`4` = list(0L), `5` = list(1L), `6` = list(2L), `7` = list(3L),
                 `8` = list(4L, 5L), `9` = list(7L, 6L), `10` = list(8L, 9L))
  )
}

export_run <- function(mode) {
  started <- Sys.time()
  message("starting ", mode, " projection")
  input <- make_inputs(mode)
  set.seed(if (mode == "preseason") 20260825L else 20260913L)
  sim <- cfb_simulations(
    input$games, input$teams,
    compute_results = model_results, spec = input$spec,
    simulations = n_sims, playoff_seeds = 12L, sim_include = "REG", autobid = "2026",
    tiebreaker_data = list(analytics_ratings = input$analytics),
    chunks = min(max(4L, workers * 2L), n_sims), verbosity = "NONE"
  )
  template_path <- file.path(viz, paste0("playoff_", mode, ".json"))
  template <- if (file.exists(template_path)) read_json(template_path) else list(teams = list())
  old <- setNames(template$teams %||% list(), vapply(template$teams %||% list(), function(x) x$team, character(1)))
  standings <- select_and_play(sim$standings, input)
  max_exit <- 5L
  rows <- lapply(fbs, function(team) {
    st <- standings[standings$team == team, ]
    base <- old[[team]] %||% list(team = team, conference = team_meta[[team]]$conference)
    seeds <- vapply(seq_len(12), function(seed) event_rate(st$seed == seed), numeric(1))
    base$team <- team
    base$conference <- team_meta[[team]]$conference %||% "FBS Independents"
    base$power_rank <- input$rankings$rank[match(team, input$rankings$team)]
    base$rating <- round(input$analytics$rating[match(team, input$analytics$team)], 3)
    base$avg_wins <- round(mean(st$wins), 2)
    base$avg_losses <- round(mean(st$losses), 2)
    base$conf_champ <- round(event_rate(st$conf_champ), 4)
    base$playoff <- round(event_rate(!is.na(st$seed)), 4)
    base$bye <- round(event_rate(!is.na(st$seed) & st$seed <= 4), 4)
    base$qf <- round(event_rate(st$exit >= 2), 4)
    base$sf <- round(event_rate(st$exit >= 3), 4)
    base$final <- round(event_rate(st$exit >= 4), 4)
    base$champ <- round(event_rate(st$exit == max_exit), 4)
    base$seeds <- as.list(round(seeds, 4))
    base
  })
  rows <- rows[order(-vapply(rows, `[[`, numeric(1), "playoff"), -vapply(rows, `[[`, numeric(1), "champ"))]
  max_wins <- max(standings$wins)
  win_dist <- setNames(lapply(fbs, function(team) {
    counts <- tabulate(standings$wins[standings$team == team] + 1L, nbins = max_wins + 1L)
    as.list(as.integer(counts))
  }), fbs)
  # Preserve the scenario-builder configuration fields carried by the existing
  # export, then replace every projection result with cfbseedR output.
  out <- template
  out$n_sims <- n_sims
  out$season <- 2026L
  out$version <- mode
  out$locked <- mode == "preseason"
  out$engine <- paste0("cfbseedR::cfb_simulations() v", sim$sim_params$cfbseedR_version, " regular season + committee-proxy selection")
  out$basis <- if (mode == "preseason") "Week-0 ensemble and Preseason Power Ratings; every game simulated" else "Current ensemble and Power Ratings; completed results locked"
  out$rules <- "12 teams; auto: ACC/Big12/BigTen/SEC champs + highest-ranked G6 team; 7 at-large; straight seeding, top-4 byes; first round at higher seed; fixed bracket"
  out$committee_proxy <- sprintf(
    "Ranked inside each simulation on its own record: %.2f*win_pct + %.2f*power_z + %.2f*sos_z + %.2f*power_conf (h2h-free fit on 2014-2025 committee rankings)",
    W_WINPCT, W_RATING, W_SOS, W_P4)
  out$teams <- rows
  out$bracket <- build_bracket(rows)
  out$tossups <- list()
  out$win_dist <- win_dist
  out$record_odds <- playoff_record_odds(standings, sim$games, input$games, fbs)
  out$game_impacts <- if (mode == "current") playoff_game_impacts(
    standings, sim$games, input$games, fbs, metadata = schedule) else NULL
  write_json(out, template_path, auto_unbox = TRUE, pretty = TRUE, digits = 8, null = "null")
  message("wrote ", template_path, " in ",
          round(as.numeric(difftime(Sys.time(), started, units = "mins")), 1), " minutes")
}

if (mode_arg %in% c("both", "preseason")) export_run("preseason")
if (mode_arg %in% c("both", "current")) export_run("current")
future::plan(future::sequential)
