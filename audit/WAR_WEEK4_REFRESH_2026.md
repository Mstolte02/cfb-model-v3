# Week 4 player-WAR refresh status (28 September 2026)

The 2026 regular-season Week 4 schedule is complete. The production in-season WAR
updater requires ten cumulative PFF reports covering weeks 1–4. Eight were staged
under `../source-data/pff_api/war_windows/2026_w01-04/`; the cumulative
`rushing.csv` and `pass_rush.csv` requests returned PFF upstream HTTP 500 on two
separate four-attempt runs. A Week 4-only rushing request succeeded, but it cannot
replace a cumulative report in the validated updater: PFF grades and rates are not
known to aggregate linearly across windows.

**No partial Week 4 WAR was published.** `viz/data/players_inseason.json` remains
through Week 3 and retained SHA-256
`1279182820CD01A88098B03A7B1EE046B2F642BB820C956E78C24D760CA92D0E`.
The preseason 2026 WAR still covers all 5,901 two-deep players; the Week 3
in-season overlay matched 4,672 of them to charted PFF players. Uncharted players
retain their preseason projection rather than receiving an invented update.

`scripts.update_inseason_war` now stages a full refresh before replacing any cached
reports, and refuses to touch the published payload when one is missing. A test
exercises the failure path. Once the two API-compatible cumulative reports are
available, complete the Week 1–4 window and run:

```powershell
venv\Scripts\python -m scripts.sync_pff_war_windows --seasons 2026 --windows 1-4
venv\Scripts\python -m scripts.update_inseason_war --week 4 --skip-pull
venv\Scripts\python -m unittest tests.test_inseason_war -v
```

The first command skips the eight reports already staged and retries the two
missing ones. The second updates the player display and team cutoff payload only
after the ten-file window is complete.

## Resolution (4 October 2026)

The next cumulative refresh succeeded through Week 5. All ten PFF reports for
weeks 1–5 were downloaded, including `rushing.csv` and `pass_rush.csv`, and the
atomic updater replaced the cached window only after confirming that the set was
complete.

`viz/data/players_inseason.json` now reports Week 5 and matches 4,898 of 5,901
two-deep players to 7,570 charted PFF players. Its SHA-256 is
`4851F34925053CAF70BA7F2B743772BCC7153C6C020E3791DB4309B517E1278E`.
The Week 3 team-WAR cutoff was rebuilt, PFF team form was extended through Week 5,
all 271 completed FBS-vs-FBS games were replayed, and the published power ratings
and current playoff simulation were regenerated from that state.
