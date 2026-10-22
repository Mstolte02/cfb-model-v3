# Most Deserving ranking

Most Deserving ranks earned results using the Colley matrix method, a
schedule-adjusted win/loss rating. It has no supervised training target and does
not fit or imitate CFP committee rankings. The former fitted weights and fitter
have been removed. Power Ratings and playoff projections are separate models.

## Eligible results

Use completed, scored regular-season games with both teams in the published FBS
universe (`ratings.teams`). FCS games are excluded from records, schedule strength,
and every calculation. Teams enter after their first eligible game. Ties count
as half a win and half a loss. Repeat opponents count once for each played game.

## Calculation

For each team i, let G be eligible games, W wins, L losses and T ties. Solve all
team ratings simultaneously:

    (G_i + 2) r_i - sum(r_opponent for every eligible game) = 1 + (W_i - L_i)/2

The diagonal of the matrix is G + 2; off-diagonal entries are minus the number
of games against that opponent. The +2 gives every team the same one-win,
one-loss starting point. All teams begin at .500, independent of conference,
preseason expectations or prior seasons.

The displayed score can be decomposed exactly:

    record base = 100 * (W + T/2 + 1) / (G + 2)
    SOS = average of opponents' solved ratings, counting each game
    SOS adjustment = 100 * G / (G + 2) * (SOS - .500)
    resume score = record base + SOS adjustment = 100 * r

SOS rank 1 is the toughest schedule already played. SOS is already incorporated
in the score and must not be added a second time. A difficult loss still hurts
relative to winning the same game; playing stronger teams raises schedule credit.

This is a percentage-like **index**, not a calibrated future win probability.
Colley ratings are not constrained to [0,1]; do not clamp scores or label them as
forecast percentages. Small samples and disconnected schedules limit comparisons.
Scores within 1e-8 displayed score points share a competition rank (1, 1, 3).
Alphabetical order only arranges tied teams on screen. All teams tied at rank 25
are displayed. Predictive power rankings do not break ties.

Conference labels, polls, scoring margins, venues, roster what-if state and
preseason ratings have no influence. Head-to-head results enter as ordinary
wins/losses; there is no extra arbitrary head-to-head bonus.

Reference: Wesley Colley, *The Colley Matrix Explained*,
https://www.colleyrankings.com/matrate.pdf

Validation: `node --test tests/test_ranking_history.cjs` covers exact solutions,
SOS credit for otherwise identical records, score decomposition, result reversal,
shared ranks, repeat opponents, ties, and independence from conference, predictive
ratings, margins, venues, FCS results, postseason and unfinished games.

## Stock Watch

Stock Watch compares the latest two published `ratings.history` snapshots for the
weekly view and the explicitly labeled preseason snapshot for the season view.
Missing baselines produce an empty state. Teams sort by rank places gained or lost,
with neutral-win-rate percentage-point changes shown alongside rank movement.
