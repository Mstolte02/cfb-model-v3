suppressPackageStartupMessages(library(cfbseedR))
source('scripts/conference_titles.R')
teams <- data.frame(team = c('East A','East B','West A','West B'), conference = 'SEC',
                    division = 'FBS', postseason_eligible = TRUE)
# Head-to-head, not model power, separates teams tied in conference record.
games <- data.frame(sim = 1L, game_type = 'REG', week = c(1L,2L,3L,4L),
  home_team = c('East A','West A','East A','East B'),
  away_team = c('East B','West B','West A','West B'), result = c(7,7,-7,7),
  home_points = c(28,28,21,28), away_points = c(21,21,28,21), neutral = 0L,
  conference_game = c(TRUE,TRUE,TRUE,TRUE))
analytics <- data.frame(team = teams$team, rating = c(1,4,3,2))
# Force the second finalist to win. A regular-season #1 is NOT automatically champion.
force_upset <- function(teams,games,week_num,spec) {
  games$result <- -7
  list(teams=teams,games=games)
}
out <- simulate_conference_titles(games,teams,games,analytics,force_upset,NULL)
title <- out$games[out$games$game_type=='CONF_CHAMP',]
stopifnot(nrow(title)==1, title$home_team=='West A', title$away_team=='East A',
          out$standings$team[out$standings$conf_champ]=='East A')
champ <- out$standings[out$standings$team=='East A',]
stopifnot(champ$wins==2,champ$losses==1,champ$conf_wins==1,champ$conf_losses==1)
# An explicitly non-conference same-league game must not affect the race.
extra <- games[1,];extra$week<-5L;extra$result<- -21;extra$conference_game<-FALSE
with_extra <- rbind(games,extra)
st <- conference_standings(with_extra,teams,with_extra,analytics)
stopifnot(st$conf_wins[st$team=='East A']==1,st$losses[st$team=='East A']==2)
# Sun Belt division opt-in is present with correct 2026 membership.
sun <- conference_teams(data.frame(team=c('App State','James Madison','Louisiana Tech','Troy'),conference='Sun Belt'))
stopifnot(identical(sun$conf_division,c('East','East','West','West')))
cat('Conference title upset, head-to-head, nonconference exclusion and Sun Belt divisions passed\n')

# Chunking preserves simulation IDs and each draw's standings across the boundary.
many <- do.call(rbind, lapply(seq_len(101), function(id) {
  draw <- games; draw$sim <- id; draw
}))
batched <- conference_standings(many, teams, games, analytics)
stopifnot(nrow(batched) == 404L, length(unique(batched$sim)) == 101L,
          all(batched$conf_rank[batched$team == 'West A'] == 1L),
          all(batched$conf_rank[batched$team == 'East A'] == 2L),
          all(tapply(batched$conf_champ, batched$sim, sum) == 1L))
cat('Batched conference standings preserve all 101 independent draws\n')
