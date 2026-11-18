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
