"""Data-source registry for Fourth & Jev enrichment."""
DATA_SOURCES = {
    "existing_model": {"priority": 0, "status": "wired", "temporal_rule": "outer-fold / prior-week only"},
    "cfbd_game_advanced": {"priority": 0, "status": "wired", "temporal_rule": "weeks strictly before game week"},
    "pff_weekly_team": {"priority": 0, "status": "wired_optional", "temporal_rule": "through prior week"},
    "projected_war": {"priority": 0, "status": "wired", "temporal_rule": "preseason projection only"},
    "recruiting_returning": {"priority": 0, "status": "wired", "temporal_rule": "preseason-known"},
    "availability": {"priority": 1, "status": "needed", "temporal_rule": "status timestamp <= kickoff"},
    "qb_status": {"priority": 1, "status": "needed", "temporal_rule": "status timestamp <= kickoff"},
    "roster_churn": {"priority": 1, "status": "needed", "temporal_rule": "transaction/publication timestamp <= kickoff"},
    "weather": {"priority": 1, "status": "needed", "temporal_rule": "forecast/observation vintage <= kickoff decision time"},
    "rest_travel": {"priority": 1, "status": "partial", "temporal_rule": "schedule known before kickoff"},
    "coaching_continuity": {"priority": 2, "status": "available_in_repo_research", "temporal_rule": "preseason/known changes"},
}
