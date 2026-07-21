"""In-season Elo layer (methodology §5.3). Seed each team's rating from the
preseason model, then update after every game using margin-of-victory, so ratings
evolve as the season reveals who was mis-rated (the static-preseason limitation the
diagnostics flagged via team residual clustering).

Seed:    R = 1500 + 400*log10(p/(1-p)), p = model's win-prob vs an average team.
Expect:  E_home = 1 / (1 + 10^(-(R_home - R_away + HFA)/400))
Update:  R += K * MOV(margin, elo_gap) * (S - E)
MOV multiplier (538-style): damped for large favorites so blowouts don't overshoot.
"""
from __future__ import annotations

import numpy as np


def to_elo(p, base=1500.0, scale=400.0):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return base + scale * np.log10(p / (1 - p))


def expected(r_home, r_away, hfa):
    return 1.0 / (1.0 + 10 ** (-((r_home - r_away + hfa) / 400.0)))


def mov_multiplier(margin, elo_gap_signed):
    """Bigger for larger margins, damped when the favorite was already heavy."""
    return np.log(abs(margin) + 1.0) * (2.2 / (abs(elo_gap_signed) * 0.001 + 2.2))


def update(ratings, home, away, home_pts, away_pts, hfa, k=40.0, neutral=False):
    """Update ratings in place for one game; return the pregame expected home prob."""
    rh, ra = ratings[home], ratings[away]
    h = 0.0 if neutral else hfa
    e = expected(rh, ra, h)
    margin = home_pts - away_pts
    s = 1.0 if margin > 0 else 0.0
    m = mov_multiplier(margin, rh - ra + h)
    delta = k * m * (s - e)
    ratings[home] = rh + delta
    ratings[away] = ra - delta
    return e
