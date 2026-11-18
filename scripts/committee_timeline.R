# Weekly CFP rank transition trained on dated polls and results from 2014-25.
# Each simulation moves one week at a time. Rank points are 31 - projected rank.
committee_pod_order <- function(score, wins, bonus) {
  remaining <- names(score)
  carry <- character()
  placed <- character()
  while (length(remaining)) {
    available <- setdiff(remaining, carry)
    available <- available[order(score[available], available)]
    pool <- c(carry, head(available, max(0L, 6L - length(carry))))
    adjusted <- vapply(pool, function(t) {
      score[[t]] - bonus * sum(vapply(setdiff(pool, t), function(other)
        paste(t, other, sep = '\r') %in% wins, logical(1)))
    }, numeric(1))
    ordered <- pool[order(adjusted, score[pool], pool)]
    take <- min(if (length(placed) < 16L) 4L else 3L, length(ordered))
    placed <- c(placed, head(ordered, take))
    carry <- tail(ordered, -take)
    remaining <- setdiff(remaining, head(ordered, take))
  }
  placed
}

committee_timeline_scores <- function(standings, games, poll, model, fbs, conf) {
  stopifnot(identical(as.integer(model$version), 1L), length(poll$rankings) == 25L)
  weights <- unlist(model$weights)
  official <- setNames(vapply(poll$rankings, function(x) as.integer(x$rank), integer(1)),
                       vapply(poll$rankings, `[[`, character(1), 'team'))
  cutoff <- as.Date(poll$games_through)
  first_week <- max(as.integer(vapply(schedule, function(x)
    if (as.Date(substr(x$d, 1, 10)) <= cutoff) as.integer(x$w) else 0L, integer(1))))
  stopifnot(all(names(official) %in% fbs))
  power <- setNames(conf %in% P4_CONFS | names(conf) == 'Notre Dame', names(conf))
  scores <- numeric(nrow(standings))
  ids <- unique(standings$sim)
  game_ix <- split(seq_len(nrow(games)), games$sim)
  standing_ix <- split(seq_len(nrow(standings)), standings$sim)
  for (sim in ids) {
    gs <- games[game_ix[[as.character(sim)]], ]
    gs <- gs[gs$game_type %in% c('REG', 'CONF_CHAMP'), ]
    rank <- setNames(rep(30, length(fbs)), fbs)
    rank[names(official)] <- official
    weeks <- sort(unique(gs$week[gs$week > first_week]))
    for (week in weeks) {
      played <- gs[gs$week <= week, ]
      participants <- unique(c(played$home_team, played$away_team))
      win <- setNames(rep(0, length(participants)), participants)
      total <- setNames(rep(0, length(participants)), participants)
      for (i in seq_len(nrow(played))) {
        h <- played$home_team[i]; a <- played$away_team[i]
        total[c(h, a)] <- total[c(h, a)] + 1
        winner <- if (played$result[i] > 0) h else a
        win[winner] <- win[winner] + 1
      }
      pct <- win / pmax(total, 1)
      wins <- ifelse(played$result > 0, played$home_team, played$away_team)
      losses <- ifelse(played$result > 0, played$away_team, played$home_team)
      head_to_head <- paste(wins, losses, sep = '\r')
      predicted <- setNames(numeric(length(fbs)), fbs)
      for (team in fbs) {
        home <- which(played$home_team == team)
        away <- which(played$away_team == team)
        ix <- c(home, away)
        opp <- c(played$away_team[home], played$home_team[away])
        won <- c(played$result[home] > 0, played$result[away] < 0)
        quality <- unname(pct[opp]); quality[is.na(quality)] <- .5
        recent <- played$week[ix] > if (week == min(weeks)) first_week else prev_week
        title <- played$game_type[ix] == 'CONF_CHAMP'
        ordinary <- recent & !title
        values <- c(prior_rank=rank[[team]], win_pct=mean(won),
                    sos=if (length(quality)) mean(quality) else 0,
                    quality_wins=sum(quality[won]), bad_losses=sum(1-quality[!won]),
                    recent_win=sum(won & ordinary), recent_loss=sum(!won & ordinary),
                    recent_win_quality=sum(quality[won & ordinary]),
                    recent_loss_badness=sum(1-quality[!won & ordinary]),
                    title_win=sum(won & recent & title),
                    title_loss=sum(!won & recent & title),
                    power=as.numeric(power[[team]]), games=length(ix))
        if (!length(ix)) values['win_pct'] <- 0
        predicted[[team]] <- model$intercept + sum(weights[names(values)] * values)
      }
      ordered <- committee_pod_order(predicted, head_to_head,
                                     as.numeric(model$pod_head_to_head_bonus))
      full_rank <- setNames(seq_along(ordered), ordered)
      rank <- pmin(full_rank, 30)
      prev_week <- week
    }
    ix <- standing_ix[[as.character(sim)]]
    final_rank <- if (length(weeks)) full_rank else rank
    scores[ix] <- 31 - unname(final_rank[standings$team[ix]])
  }
  list(score=scores, official=official,
       method='Weekly as-of CFP transitions fitted on 2014-25; six-team comparison pools, four then three placements')
}
