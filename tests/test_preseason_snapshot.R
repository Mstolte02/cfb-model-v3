# Extract production functions without loading cfbseedR or running season draws.
for (e in parse('scripts/simulate_playoff.R')) {
  if (is.call(e) && identical(e[[1]], as.name('<-')) && is.symbol(e[[2]]) &&
      as.character(e[[2]]) %in% c('resolve_snapshot_bracket', 'export_run')) eval(e)
}
# Exercise the real eleven-game feed topology, including first-round upsets.
p <- jsonlite::fromJSON('viz/data/playoff_preseason.json', simplifyVector = FALSE)
# Clear advancement slots if the on-disk snapshot is already resolved.
for (i in 5:8) p$bracket$games[[i]]$bottom <- NULL
for (i in 9:11) { p$bracket$games[[i]]$top <- NULL; p$bracket$games[[i]]$bottom <- NULL }
calls <- 0L
probability <- function(home, away, neutral, spec) {
  stopifnot(identical(spec, 'week-zero'), length(home) == 1L, length(away) == 1L, home != away)
  calls <<- calls + 1L
  if (neutral) .75 else .25
}
b <- resolve_snapshot_bracket(p$bracket, 'week-zero')
stopifnot(calls == 11L)
for (i in seq_along(b$games)) {
  g <- b$games[[i]]
  stopifnot(g$p %in% c(.25, .75), g$winner == if (g$p >= .5) g$top else g$bottom)
  src <- b$feeds[[as.character(i - 1L)]]
  if (length(src)) {
    winners <- vapply(src, function(k) b$games[[k + 1L]]$winner, character(1))
    stopifnot(all(winners %in% c(g$top, g$bottom)))
  }
}
# Future refreshes must leave a completed snapshot byte-for-byte untouched.
viz <- tempfile('preseason-lock-')
dir.create(viz)
path <- file.path(viz, 'playoff_preseason.json')
jsonlite::write_json(list(season = 2026L, locked = TRUE, snapshot_lock = list(version = 1L)), path, auto_unbox = TRUE)
read_json <- function(path) jsonlite::fromJSON(path, simplifyVector = FALSE)
args <- character()
make_inputs <- function(mode) stop('simulation entered', call. = FALSE)
before <- readBin(path, 'raw', n = file.info(path)$size)
export_run('preseason')
stopifnot(identical(before, readBin(path, 'raw', n = file.info(path)$size)))
# Explicit rebuilding, an incomplete snapshot, and current mode still run normally.
args <- '--rebuild-preseason'
stopifnot(grepl('simulation entered', as.character(try(export_run('preseason'), silent = TRUE))))
args <- character()
stopifnot(grepl('simulation entered', as.character(try(export_run('current'), silent = TRUE))))
jsonlite::write_json(list(season = 2026L, locked = TRUE), path, auto_unbox = TRUE)
stopifnot(grepl('simulation entered', as.character(try(export_run('preseason'), silent = TRUE))))
unlink(viz, recursive = TRUE)
cat('Preseason snapshot: frozen week-zero path, advancement, immutable refresh and explicit rebuild passed\n')
