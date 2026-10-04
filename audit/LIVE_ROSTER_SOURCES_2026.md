# Live roster and availability sources (2026)

## Production rule

- **Current roles and starters:** PFF weekly offense and defense player reports. The
  reports provide participation and total snaps; the live roster assigns the normal
  first-unit count at each position to the busiest currently available players. The
  raw weekly files are staged outside Git under
  `source-data/pff_api/player_weekly/` because the licensed data cannot be published.
- **Confirmed absences:** append-only team announcements in
  `war_model/availability_events_2026.csv`. An `out` event removes the player's WAR
  from the current player payload and the newest team WAR cutoff. It does not rewrite
  the preseason roster or any earlier cutoff.
- **Fallback:** if PFF has not charted enough players in a position room, retain the
  preseason depth-chart starter flags. This avoids guessing from a partial sample.

## Source audit

The [PFF API reference](https://developer.pff.com/reference/) exposes team rosters,
player statistics, grades, and snap participation, but no supported injury-report or
current depth-chart operation. The [CollegeFootballData API
overview](https://api.collegefootballdata.com/getting-started) covers game, team,
player, recruiting, ranking, and analytics data but likewise documents no injury
feed. Therefore neither is treated as an authoritative injury source.

For Notre Dame, the September 28 update is tied to the school's official [Marcus
Freeman weekly press-conference
page](https://fightingirish.com/news/2026/09/28/north-carolina-marcus-freeman-weekly-press-conference-92826-notre-dame-football).
The resulting event marks Quincy Porter out and removes his remaining current WAR;
PFF participation determines the receivers who replace him in the live first unit.

## Refresh commands

```text
python -m scripts.sync_pff_weekly_players --seasons 2026 --weeks 1-5
python -m war_model.materialize_availability
python -m scripts.update_inseason_war --week 5
```

Advance the final week in both PFF commands after a slate is complete. Add injury
events only when a team, conference, or comparably direct source confirms the status.
