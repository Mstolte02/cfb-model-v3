"""Render the evaluated WAR decision without refitting or selecting a winner."""
import json
from config import ROOT

DECISION=("Keep the current production model unchanged. Rebuilt in-season WAR has mixed, "
          "uncertain incremental value; these results establish neither a reliable gain nor a reason "
          "to remove it. Do not promote the tested interactions. Retain the passing–coverage gap "
          "and after-contact–tackling product as research leads with their linear controls.")


def write_report():
    path=ROOT/'audit/output/chronological_war_matchups.json'
    result=json.loads(path.read_text());result['decision']=DECISION
    # Public source fingerprints contain no licensed values or private paths.
    manifests={}
    for year in (2022,2023,2024,2025):
        manifest=ROOT/f'artifacts/chronological_war/{year}/weights_manifest.json'
        if manifest.exists():manifests[str(year)]=json.loads(manifest.read_text())
    if manifests:result['war_fit_manifests']=manifests
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    summary=ROOT/'viz/power-xray/war-research.json'
    data=json.loads(summary.read_text());data['decision']=DECISION
    summary.write_text(json.dumps(data,allow_nan=False,separators=(',',':'))+'\n')
    rows=[];seasons=[]
    for name,report in result['metrics'].items():
        m=report['overall']
        rows.append(f"| {name} | {m['brier']:.6f} | {m['log_loss']:.6f} | {m['auc']:.6f} | {m['accuracy']:.2%} | {m['calibration_intercept']:.4f} | {m['calibration_slope']:.4f} | {m['ece']:.2%} | {m['confident_error_rate']:.2%} ({m['confident_games']}) |")
        s=report['by_season']
        seasons.append(f"| {name} | {s['2024']['brier']:.6f} | {s['2025']['brier']:.6f} | {s['2024']['log_loss']:.6f} | {s['2025']['log_loss']:.6f} |")
    text=f'''# Chronological in-season WAR and opponent interactions

Decision recorded October 12, 2026. {DECISION}

## What was rebuilt

The complete **in-season PFF facet-WAR measurement and calibration chain** was rebuilt
in isolated directories for 2022–2025. For each target season, earlier seasons alone
fit facet consolidation/PCA, reliability, facet weights, team scales and WAA conversion.
Raw player measurements use earlier-season weighted moments; the target year does not
normalize its own inputs. Player Kalman priors and the conversion to WAR use earlier
years. Opponent strength for an observed window uses games through that window's cut,
and postseason rows with reset week numbers are excluded.

Player windows end at weeks 3, 6 and 9. Completed later windows supply calibration
labels only for subsequent seasons. 2022 supplies the first calibration history;
2023 fits its rule on 2022, 2024 on 2022–2023, and 2025 on 2022–2024. Team histories
contain 402, 402 and 408 team/cut rows respectively. A cut becomes eligible only
when game week is strictly greater than the cut. There are no future-year
leave-one-season-out rules or inherited full-history normalization in this update.

**Scope limit:** the four members' original preseason `war_projected` and
`player_prod_war` inputs retain the recovered expanded-WAR production lineage.
Their earlier global upstream transforms were not rebuilt here. This is not a claim
that every input in the entire production model is now chronologically clean.
The comparison isolates the rebuilt in-season update conditional on those shared
preseason inputs. A fully end-to-end preseason WAR audit remains a dependency for
stronger production claims.

## Evaluation contract

All 25 candidates use identical games and four equally weighted member probabilities.
Preseason member coefficients are refitted on earlier years using the previously
saved earlier-year selections. Unit scaling, logistic coefficients and regularization
are fitted without outer test labels. Each stack selects C from
0.01, 0.03, 0.1, 0.3 and 1 by earlier chronological validation log loss.
The first inner split is weeks before 9 versus weeks 9 onward when only one year is
available. This selection is repeated equally for every candidate and member.

There are 729 warmup predictions in 2023, then **1,460 evaluated games**:
741 in 2024 and 719 in 2025. The team stack needs 2023's earlier-calibrated WAR to
learn a nonzero WAR coefficient, so 2023 is not credited as independent WAR evidence.

The `no_inseason_war` control still includes preseason WAR. It is a newly tuned,
matched ablation of the lean stack, not an exact replay of the already published
2026 production model or its old fitted penalties. In this research all games use
the same lean columns; absent PFF offense is zero-imputed rather than switching
to production's no-PFF base stack. This differs from the prior historical runtime
reconciliation and explains why its baseline score is slightly different.

## Opposing units

Six predefined unit pairings are tested: passing EPA versus coverage EPA allowed;
rushing EPA versus rushing EPA allowed; pressure allowed versus pressure generated;
yards before contact versus yards before contact allowed; explosive pass rate versus
explosive pass rate allowed; yards after contact versus yards after contact allowed.
Defensive signs are oriented so higher means stronger. Every feature uses the exact
prior-week PFF export. Missing interactions become zero; missing standardized main
effects use the training mean. Scales use only the earlier training observations.

Each pairing has linear O/D differences, an opposing product, and a nonlinear gap.
The product is O_home D_away − O_away D_home. The gap is
g(O_home − D_away) − g(O_away − D_home), where g(x)=x|x|.
Nonlinear candidates include their identical linear main effects. The combined
families contain all six pairs. These terms reverse sign when sides swap and preserve
complementary probabilities. Separately standardized O and D are not predicted points.

## Scores on common held-out games

Lower Brier/log loss/ECE are better. Higher AUC/accuracy are better. Calibration
intercept should approach 0 and slope 1; these are diagnostics fitted on evaluation
predictions, never used to adjust those predictions. ECE uses ten fixed bins.
Confidence errors are incorrect picks among predictions at least 80% confident;
the selected population changes across candidates, so this is not a paired causal comparison.

| Candidate | Brier | Log loss | AUC | Accuracy | Cal. intercept | Cal. slope | ECE | Confident error (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Stability

| Candidate | 2024 Brier | 2025 Brier | 2024 log loss | 2025 log loss |
|---|---:|---:|---:|---:|
{chr(10).join(seasons)}

WAR's Brier change versus no in-season WAR is −0.000052; log loss changes +0.000301.
Accuracy increases from 72.40% to 72.60%, AUC is essentially unchanged, and ECE improves
from 1.50% to 1.26%. The confident error rate moves from 10.43% to 10.49%.
WAR hurts both proper scores in 2024 and helps both in 2025.

Paired season/week bootstrap 95% intervals are [−0.000954, +0.000890] for WAR's
Brier difference and [−0.002118, +0.002825] for its log-loss difference.
Both include zero. There are 31 season/week blocks but only two season blocks;
season resampling is too sparse to supply strong confidence. Weekly breakdowns,
calibration bins, balanced accuracy, entropy, fitted coefficients, selected penalties
and all paired comparisons are in the JSON artifact.

The all-pair interaction models lose to the simple WAR model and to their all-pair
linear control on pooled Brier and log loss. The targeted passing/coverage gap
improves on its matching linear control (ΔBrier −0.000249, Δlog loss −0.000805),
but intervals include zero and season-level performance is mixed.
The after-contact/tackling product improves on its matching linear control in
both seasons (pooled ΔBrier −0.000263, Δlog loss −0.000757). Its paired log-loss
interval [−0.001330, −0.000174] excludes zero, but its Brier interval includes zero
and it still loses to the simple WAR model overall. This is a research lead,
not a production winner. Twenty-five inspected candidates and uncorrected paired
intervals cannot establish confirmatory significance; cross-week team dependence
also remains. The targeted pair tests followed inspection of the combined-family
results, so they are explicitly exploratory.

## Production and X-Ray

No production model, team ratings, locked bets, or immutable prediction rows were
rewritten. Existing in-season WAR remains active provisionally; this experiment
does not establish its benefit. No new interactions were promoted. Scheduled
updates continue as configured.

The X-Ray WAR lab includes all 2,189 historical games and all 25 candidates.
For each of four members it shows the exact feature × fitted-coefficient log-odds
contributions. Their sum maps through the logistic function, and the four resulting
probabilities average to the displayed ensemble prediction. JSON chunks load only
for the selected season/week. JavaScript tests verify every published prediction
and swapped-side reciprocity. The original live X-Ray remains the active-model view.

The repository PFF secret is configured and was verified by successful Actions run
[38340063670](https://github.com/Mstolte02/cfb-model-v3/actions/runs/38340063670).
Credentials and licensed raw tables are private. Retrospective downloads are not
archived as-of-kickoff source snapshots; historical injuries were not reconstructed.
CFBD play-value facets lack player-window sources and are omitted from this update.

## Reproduce

With the recovered private source tree and Python dependencies installed:

```shell
python -m scripts.chronological_war_rebuild
python -m scripts.war_matchup_chronological
python -m scripts.report_chronological_war
python -m unittest tests.test_chronological_war tests.test_inseason_war tests.test_historical_ensemble_research tests.test_ensemble_blend_research tests.test_pff_carry_forward
node --test tests/test_market_board.cjs tests/test_power_xray.cjs tests/test_war_research.cjs
```

Private fit manifests and calibration artifacts live under `artifacts/chronological_war`.
Public results: [full metrics and provenance](output/chronological_war_matchups.json),
[game predictions](output/chronological_war_matchup_predictions.csv), and
[X-Ray](https://mstolte02.github.io/cfb-model-v3/power-xray/war-research.html).
'''
    (ROOT/'audit/CHRONOLOGICAL_WAR_MATCHUPS.md').write_text(text,encoding='utf-8')


if __name__=='__main__':write_report()
