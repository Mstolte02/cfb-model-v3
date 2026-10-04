# Live roster and availability sources (2026)

## Production rule

- **Current roles and starters:** PFF weekly offense and defense player reports. The
  reports provide participation and total snaps; the live roster assigns the normal
  first-unit count at each position to the busiest currently available players. The
  raw weekly files are staged outside Git under
  `source-data/pff_api/player_weekly/` because the licensed data cannot be published.
- **Injury reports:** `scripts/sync_injury_reports.py` reads three public reports and
  appends status changes to `war_model/availability_events_2026.csv` as
  `source_type=injury_report`:
  - [MyGameSim](https://www.mygamesim.com/cfb/college-football-injuries.asp): every
    FBS team, full names. The backbone; the sync stops if it returns under 100 rows.
  - [Covers](https://www.covers.com/sport/football/ncaaf/injuries): initial + last
    name, dated status and a note. Corroborates and adds players.
  - [Rotowire](https://www.rotowire.com/cfootball/injury-report.php): the free table
    returns only seven rows, so it adds very little.

  Each source's status maps to an expected share of the next game (out 0, doubtful
  0.25, questionable 0.5, probable 1). Sources that disagree (about one shared player
  in four) are averaged, so WAR stays continuous. A player who drops off every report
  gets a `clear` event. Players not on the projected two-deep are ignored: the model
  holds no WAR for them.
- **Confirmed absences:** team announcements and manual audits, also in the event
  stream. A public report never clears or softens one of these.
- **How absences enter the model:** the player's current WAR is scaled by his share.
  The team signal D is WAR *per week*, so the lost season WAR is divided by 12 before
  it joins the newest cutoff. Earlier cutoffs, the preseason roster and completed
  games are never rewritten.
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
python -m scripts.sync_injury_reports
python -m war_model.materialize_availability
python -m scripts.update_inseason_war --week 5
python -m scripts.update_v4
```

Advance the final week in both PFF commands after a slate is complete. Add injury
events only when a team, conference, or comparably direct source confirms the status.
