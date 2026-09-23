"""Predeclared Jev question families."""
from __future__ import annotations


def football_questions() -> dict:
    return {
        "home_win": {"type": "noul", "instructions": "Will the home team win the game outright?", "criteria": {"true": "Home team wins", "false": "Away team wins"}},
        "home_by_7_plus": {"type": "noul", "instructions": "Will the home team win by at least 7 points?"},
        "home_by_14_plus": {"type": "noul", "instructions": "Will the home team win by at least 14 points?"},
        "home_by_21_plus": {"type": "noul", "instructions": "Will the home team win by at least 21 points?"},
        "away_by_7_plus": {"type": "noul", "instructions": "Will the away team win by at least 7 points?"},
        "within_3": {"type": "noul", "instructions": "Will the final scoring margin be 3 points or fewer?"},
        "tail_shape": {"type": "choice", "instructions": "Which description best fits the likely shape of the game outcome distribution?", "criteria": {"balanced": "Roughly symmetric uncertainty around the central outcome.", "home_right_tail": "Elevated chance of a large home win relative to the central expectation.", "away_right_tail": "Elevated chance of a large away win relative to the central expectation.", "two_sided_fat_tails": "Both teams have unusually plausible paths to decisive wins.", "compressed": "Outcomes are unusually concentrated near the central expectation."}},
        "variance_level": {"type": "score", "instructions": "Rate how intrinsically high-variance this matchup is before considering betting prices.", "criteria": ["Very low variance", "Low variance", "Typical variance", "High variance", "Very high variance"]},
        "upset_path_strength": {"type": "score", "instructions": "Rate how strong the underdog's concrete football mechanisms are for producing an upset.", "criteria": ["No credible upset mechanism", "Weak", "Plausible", "Strong", "Multiple strong upset mechanisms"]}
    }


def market_questions() -> dict:
    return {
        "market_mispricing": {"type": "noul", "instructions": "Is the supplied market price materially inconsistent with the supplied football probability evidence after accounting for uncertainty and vig?"},
        "tail_mispricing": {"type": "noul", "instructions": "Does the market appear to underprice a specific tail outcome supported by the football evidence?"},
        "edge_quality": {"type": "score", "instructions": "Rate the quality of the apparent edge, emphasizing robustness rather than raw disagreement.", "criteria": ["None", "Weak/noisy", "Interesting", "Strong", "Exceptional and robust"]}
    }
