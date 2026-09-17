# Documentation

This folder holds the long-form material that used to sit in the root README. The
root [README](../README.md) tells you what the model is and how to run it. Everything
below is the detail behind it.

## Release notes

The full history of the model, one file per major version. Start with
[the changelog index](changelog/README.md).

| File | Covers |
|---|---|
| [changelog/v4.md](changelog/v4.md) | v4, the model that ships today |
| [changelog/v3.md](changelog/v3.md) | v3.0 to v3.10 — player WAR and the web app |
| [changelog/v2.md](changelog/v2.md) | v2 — the margin ensemble and the playoff simulation |
| [changelog/v1.md](changelog/v1.md) | v1 — the original README, kept as written |

## Experiments and audits

Each file in [`audit/`](../audit/) records one question, the test that answered it,
and the decision. A feature ships only if it clears the adoption bar in consecutive
selection windows. Most of these documents record a rejection, and that is the point:
the evidence stays in the repository whether the answer was yes or no.

### The model as a whole

| Document | Question |
|---|---|
| [CFB_MODEL_V3_AUDIT.md](../audit/CFB_MODEL_V3_AUDIT.md) | The adversarial audit that produced v4 |
| [STANDARDISATION_AND_COLLINEARITY.md](../audit/STANDARDISATION_AND_COLLINEARITY.md) | The feature set cut from fifteen columns to seven |
| [RATING_ARCHITECTURE_EXPERIMENTS.md](../audit/RATING_ARCHITECTURE_EXPERIMENTS.md) | Does a feature belong inside the rating, or beside it? |
| [INSEASON_UPDATE_EXPERIMENTS.md](../audit/INSEASON_UPDATE_EXPERIMENTS.md) | The weekly update rule |

### Features that were tested

| Document | Question |
|---|---|
| [COACH_EFFECTS_EXPERIMENTS.md](../audit/COACH_EFFECTS_EXPERIMENTS.md) | Do head-coach effects predict games? (no) |
| [DECISION_PROFILE_EXPERIMENTS.md](../audit/DECISION_PROFILE_EXPERIMENTS.md) | Do play-call tendencies predict games? (no) |
| [CONTEXT_AND_PLAYER_PROJECTION_EXPERIMENTS.md](../audit/CONTEXT_AND_PLAYER_PROJECTION_EXPERIMENTS.md) | Rest, travel, load, and the player projection |
| [CURVATURE_EXPERIMENTS.md](../audit/CURVATURE_EXPERIMENTS.md) | Is the talent effect nonlinear? |
| [FACET_MATCHUP_EXPERIMENTS.md](../audit/FACET_MATCHUP_EXPERIMENTS.md) | Unit against unit: WAR split by facet and position |
| [PLAYER_PRODUCTION_EXPERIMENTS.md](../audit/PLAYER_PRODUCTION_EXPERIMENTS.md) | The player-production forecast as a game feature |
| [TALENT_SOURCES_EXPERIMENTS.md](../audit/TALENT_SOURCES_EXPERIMENTS.md) | Positional recruiting and the rated portal (shipped) |
| [HOME_FIELD_EXPERIMENTS.md](../audit/HOME_FIELD_EXPERIMENTS.md) | Is home field team-specific? (no) |

### Player WAR

| Document | Question |
|---|---|
| [WAR_SOURCES_EXPERIMENTS.md](../audit/WAR_SOURCES_EXPERIMENTS.md) | Which sources go into WAR, and what WAR is worth alone |
| [WAR_AND_PROFITABILITY_AUDIT.md](../audit/WAR_AND_PROFITABILITY_AUDIT.md) | The role split, and intrinsic WAR for player rankings |
| [war_model/README.md](../war_model/README.md) | How the WAR build works, stage by stage |

### Markets and published numbers

| Document | Question |
|---|---|
| [MODEL_VS_MARKET_DIAGNOSIS.md](../audit/MODEL_VS_MARKET_DIAGNOSIS.md) | How far behind the market is the model, and why |
| [BET_THRESHOLD_CALIBRATION.md](../audit/BET_THRESHOLD_CALIBRATION.md) | Is any model gap large enough to bet? (not yet) |
| [TOTALS_MODEL_REPAIR.md](../audit/TOTALS_MODEL_REPAIR.md) | The repaired totals model (shipped) |
| [DYNAMIC_MARGIN_EXPERIMENTS.md](../audit/DYNAMIC_MARGIN_EXPERIMENTS.md) | Should the projected margin follow the in-season ratings? |
| [DESERVING_RANKINGS.md](../audit/DESERVING_RANKINGS.md) | The result-based résumé ranking on the site |

### Sanity checks

| Document | Question |
|---|---|
| [SEC_CONCENTRATION_AUDIT.md](../audit/SEC_CONCENTRATION_AUDIT.md) | Why the SEC held 13 of the top 25 |
| [VANDERBILT_SEC_SANITY_CHECK.md](../audit/VANDERBILT_SEC_SANITY_CHECK.md) | Why Vanderbilt rated No. 25 |
