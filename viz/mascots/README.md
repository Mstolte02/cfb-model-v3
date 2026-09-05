# Live team season cards

Open Power Rankings > Team history and choose any of the 138 rated teams.
The retro HTML/SVG card includes current rank, preseason rank movement, neutral
round-robin win rate, completed scores and a weekly trend. Print / save PDF uses
the browser's print dialog. The chart has an expandable accessible numeric table.

Both the regular export and scheduled market capture rebuild ratings.history and
ratings.game_history from published finals. Pregame probabilities are replayed
from the start-of-week state under the current model, not claimed to be archived
forecasts. Model delta is the signed natural-logit update, not Elo points. Games
against unrated opponents count in the record and display scores without invented
model probabilities. Non-final games never enter the log.

Mascots use three original built-in ImageGen sprite atlases (16 cells each), shared
by character family. These are mascot-inspired illustrations, not official mascot
portraits. Teams without a matching family use an anonymous uniformed player.
Team colors, names and logos remain specific to each program. Mapping is in
viz/team-card.js; each cell is selected in CSS without splitting the source image.

Validation:
  node --test tests/test_team_card.cjs
  python -m unittest tests.test_market_tracking tests.test_publish -q
  python scripts/publish.py

Generation prompts are recorded in viz/mascots/prompts.json.
