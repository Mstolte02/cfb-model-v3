"""Publish compact, lazy-loaded X-Ray attribution from evaluated predictions."""
import json
from collections import defaultdict
from config import ROOT


def publish(payload):
    folder=ROOT/'viz/power-xray/war-research'
    folder.mkdir(exist_ok=True)
    chunks=defaultdict(list)
    columns={}
    index=[]
    for game in payload['games']:
        key=f"{game['season']}-{game['week']:02}"
        meta={k:v for k,v in game.items() if k!='candidates'}
        index.append({**meta,'chunk':key,'index':len(chunks[key])})
        candidates={}
        for name,value in game['candidates'].items():
            columns.setdefault(name,list(value['members'][0]))
            candidates[name]={'probability':value['probability'],
                'members':[[round(m[c],12) for c in columns[name]] for m in value['members']]}
        chunks[key].append({**meta,'candidates':candidates})
    for key,rows in chunks.items():
        (folder/f'{key}.json').write_text(json.dumps(rows,allow_nan=False,separators=(',',':'))+'\n')
    summary={**payload,'games':index,'columns':columns}
    (folder.parent/'war-research.json').write_text(json.dumps(summary,allow_nan=False,separators=(',',':'))+'\n')


if __name__=='__main__':
    publish(json.loads((ROOT/'viz/power-xray/war-research.json').read_text()))
