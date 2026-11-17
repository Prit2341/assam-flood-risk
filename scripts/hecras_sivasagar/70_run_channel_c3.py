"""Run C3 through RAS Commander and score automatically after completion.

No desktop capture or progress-polling loop. Each child command is awaited.
"""
from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime,timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
label = sys.argv[1] if len(sys.argv)>1 else 'channelc3'
assert label in ['channelc3','channelc3fast']
OUT = ROOT/'digital_twin/sivasagar/results'/('channel_c3' if label == 'channelc3' else 'channel_c3fast')
assert json.loads((OUT/'build.json').read_text())['build_pass']
commands = [('compute',[sys.executable,'-u',str(HERE/'34_run_grid_project.py'),label,'432']),
            ('score',[sys.executable,'-u',str(HERE/'69_score_channel_c3.py'),label])]
for label,cmd in commands:
    print(f'Starting {label}',flush=True)
    with (OUT/f'{label}.log').open('w',encoding='utf-8') as stream:
        result = subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{label} failed; inspect {OUT/str(label+".log")}')
    print(f'Completed {label}',flush=True)
(OUT/'pipeline_complete.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),
    verdict=json.loads((OUT/'hydraulic_verdict.json').read_text())),indent=2))
print('C3 simulation and registered scoring completed.',flush=True)
