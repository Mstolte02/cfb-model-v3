# Exercise the production point-estimate constructor without running season draws.
for (e in parse('scripts/simulate_playoff.R')) {
  if (is.call(e) && identical(e[[1]], as.name('<-')) && is.symbol(e[[2]]) &&
      as.character(e[[2]]) %in% c('row_number', 'build_bracket')) eval(e)
}
teams <- c('Georgia', 'Alabama', 'Notre Dame', paste0('Team', 4:16))
rows <- lapply(seq_along(teams), function(i) list(
  team = teams[i], conference = if (i <= 2) 'SEC' else if (i == 3) 'FBS Independents'
    else c('ACC','Big 12','Big Ten','Mountain West')[1 + (i - 4) %% 4],
  predicted_resume_score = 32 - i, projected_rank = i, playoff = .99 - i / 100,
  power_rank = i, conf_champ = if (i %in% c(1,4,5,6)) 1 else 0,
  g6_bid = if (i == 7) 1 else 0, seeds = as.list(rep(0, 12))))
# Notre Dame's individual mode is #2; Alabama's is #3. Those modes must never
# override the common committee score order that places Alabama above Notre Dame.
rows[[2]]$seeds[[3]] <- .8
rows[[3]]$seeds[[2]] <- .8
b <- build_bracket(rows)
stopifnot(identical(b$seeds[1:3], c('Georgia','Alabama','Notre Dame')),
          length(unique(b$seeds)) == 12,
          all(c('Georgia','Team4','Team5','Team6','Team7') %in% b$seeds))
score <- setNames(vapply(rows, `[[`, numeric(1), 'predicted_resume_score'), teams)
stopifnot(all(diff(score[b$seeds]) <= 0))
# Reversing marginal seed frequencies cannot change the point-estimate bracket.
changed <- rows
changed[[2]]$seeds <- as.list(rep(1, 12))
changed[[3]]$seeds <- as.list(rep(0, 12))
stopifnot(identical(build_bracket(changed)$seeds, b$seeds))
# A worse projected committee score moves Alabama below Notre Dame, rather than
# leaving it glued to a modal third seed. Automatic bids still use straight seeding.
changed[[2]]$predicted_resume_score <- 20.5
lower <- build_bracket(changed)
stopifnot(match('Alabama', lower$seeds) > 3,
          match('Notre Dame', lower$seeds) < match('Alabama', lower$seeds))
rows[[1]]$predicted_resume_score <- NULL
stopifnot(inherits(try(build_bracket(rows), silent=TRUE), 'try-error'))
cat('Projected bracket: one score order, independent-mode regression, loss movement, automatic bids and missing inputs passed\n')
