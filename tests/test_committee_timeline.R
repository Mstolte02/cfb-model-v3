source('scripts/committee_timeline.R')
model <- jsonlite::fromJSON('viz/data/committee_timeline_model.json', simplifyVector=FALSE)
stopifnot(model$version == 1L,
          model$weights$recent_loss > model$weights$title_loss,
          all(vapply(model$validation, function(x) x$mae < x$prior_mae, logical(1))))
scores <- setNames(seq_len(8) + 0.1, LETTERS[1:8])
scores['B'] <- scores['A'] + .1
plain <- committee_pod_order(scores, character(), .25)
with_result <- committee_pod_order(scores, paste('B','A',sep='\r'), .25)
stopifnot(plain[1] == 'A', with_result[1] == 'B',
          identical(sort(plain), sort(with_result)))
# The actual 2026 Top 25 and the model must be complete before running draws.
poll <- jsonlite::fromJSON('viz/data/cfp_rankings.json', simplifyVector=FALSE)
stopifnot(length(poll$rankings) == 25,
          setequal(vapply(poll$rankings, `[[`, integer(1), 'rank'), 1:25),
          length(model$weights) == length(model$features))
cat('Weekly committee model: historical holdouts, title penalty, pool comparison and poll inputs passed\n')
# Test the actual simulation feature path using fixed dated results.
fbs <- vapply(poll$rankings, `[[`, character(1), 'team')
P4_CONFS <- c('ACC','Big 12','Big Ten','SEC')
conf <- setNames(rep('Big Ten',length(fbs)),fbs)
p <- poll; p$games_through <- '2026-11-30'
schedule <- list(list(d='2026-11-28',w=13L))
st <- data.frame(sim=1L,team=fbs)
gs <- data.frame(sim=1L,game_type='REG',week=13L,home_team=fbs,
                 away_team=paste0('FCS ',fbs),result=14)
winner <- fbs[8]; loser <- fbs[4]
probe <- model; probe$weights <- as.list(setNames(rep(0,length(model$features)),model$features))
probe$weights$prior_rank <- 1; probe$weights$title_margin <- -1; probe$intercept <- 0
saved_order <- committee_pod_order
committee_pod_order <- function(score,wins,bonus) { captured <<- score; names(sort(score)) }
run_margin <- function(m) {
  title <- data.frame(sim=1L,game_type='CONF_CHAMP',week=14L,
                      home_team=winner,away_team=loser,result=m)
  invisible(committee_timeline_scores(st,rbind(gs,title),p,probe,fbs,conf))
  captured
}
close <- run_margin(1); big <- run_margin(35); capped <- run_margin(70)
stopifnot(close[winner]>close[loser], big[winner]<big[loser],
          identical(big,capped), abs(big[winner]-3)<1e-10,abs(big[loser]-9)<1e-10)
probe$weights$title_margin <- 0; probe$weights$title_win_opponent <- 1
opponent <- run_margin(7)
stopifnot(abs(opponent[winner]-(8+(30-4)/10))<1e-10)
committee_pod_order <- saved_order
cat('Simulation title margin, cap and as-of opponent rank passed\n')
