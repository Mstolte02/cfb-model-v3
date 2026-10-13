"""Small historical source recovery probe. Logs only status/counts, never raw data.

Uses repository secrets inside Actions. Raw responses remain on the ephemeral
runner, and are not artifacts or commits. This probes a historical season before
attempting an expensive multi-season rebuild.
"""
import json
import os
from pathlib import Path

OUT=Path('source-recovery-status.json')

def main():
    results={'purpose':'historical source recovery probe','season':2025,'sources':{}}
    cfbd=os.environ.get('CFBD_API_KEY')
    pff=os.environ.get('PFF_API_KEY')
    # Reuse the actual clients; no credentials in subprocess arguments or output.
    if cfbd:
        from src.data import cfbd_client as C
        for label,fn in [('cfbd_games',lambda:C.games(2025)),
                         ('cfbd_game_efficiency',lambda:C.game_advanced(2025)),
                         ('cfbd_preseason_efficiency',lambda:C.advanced_season_stats(2024)),
                         ('cfbd_returning',lambda:C.returning_production(2025)),
                         ('cfbd_talent',lambda:C.talent(2025))]:
            try:
                rows=fn();results['sources'][label]={'status':'recovered' if rows else 'empty','rows':len(rows)}
            except Exception as exc:
                results['sources'][label]={'status':'failed','error_type':type(exc).__name__}
    else:results['sources']['CFBD']={'status':'credential_not_configured'}
    if pff:
        from src.data.pff_api import PffClient
        client=PffClient()
        for label,fn in [('pff_historical_directory',lambda:client.team_directory(2025)),
                         ('pff_historical_player_blocking',lambda:client.facet_report('blocking',2024)),
                         ('pff_historical_team_offense',lambda:client._request('/v2/ncaa/teams/stats',{'season':2025,'category':'offense-overall-success','weekIds':'0,1,2'}).get('rows',[])),
                         ('pff_historical_team_defense',lambda:client._request('/v2/ncaa/teams/stats',{'season':2025,'category':'defense-overall-success','weekIds':'0,1,2'}).get('rows',[]))]:
            try:
                rows=fn();n=len(rows);results['sources'][label]={'status':'recovered' if n else 'empty','rows':n}
            except Exception as exc:
                results['sources'][label]={'status':'failed','error_type':type(exc).__name__}
    else:results['sources']['PFF']={'status':'credential_not_configured'}
    results['limits']=['A successful sample proves source access, not full historical reproducibility.',
        'Bought depth-chart workbooks and unavailable historical source vintages cannot be recreated exactly from aggregate ratings.',
        'Expanded WAR lineage must be reconciled before claiming the original ensemble has been reproduced.']
    OUT.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()
