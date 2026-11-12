"""Bounded development tuning of the frozen-GRU readout; never reads test rows."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import write_json


def main():
    destination=ROOT/'docs/assets/please-trajectory-readout-study-2026-11-12.json'
    candidates=[(.01,'c001'),(.1,'c01'),(1.,'v1'),(10.,'c10'),(100.,'c100')]
    protocol={'status':'running','grid_C':[c for c,_ in candidates],
              'fit':'219 official training examples only; frozen epoch-13 GRU; no extra data',
              'selection':'Minimum validation cross entropy among PLEASE 4/4, MY 4/4 and overall >=50/60; no success if empty.',
              'development_disclosure':'Grid declared after seeing the failed C=1 attempt; these validation cases inform tuning.',
              'official_test_read':False,'candidates':[]}
    write_json(destination,protocol)
    for c,suffix in candidates:
        output=ROOT/f'models/asl-gru-fifteen-signs-trajectory-readout-{suffix}'
        report=output/'report.json'
        if not report.exists():
            log=ROOT/f'.cache/please-readout-{suffix}-20261112.log'
            with log.open('w') as stream:
                subprocess.run([sys.executable,'-u',str(ROOT/'scripts/refit_trajectory_readout.py'),
                                '--c',str(c),'--output',str(output)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        data=json.loads(report.read_text())
        row={'C':c,'checkpoint':str((output/'best.pt').relative_to(ROOT)),
             'checkpoint_sha256':data['checkpoint_sha256'],'acceptance_met':data['acceptance_met'],
             'train_correct':data['train']['correct'],'validation':data['validation'],
             'backbone_identical':data['backbone_identical'],
             'previous_weights_and_probes_unchanged':data['previous_weights_and_probes_unchanged'],
             'subset_unchanged':data['subset_unchanged'],
             'export_max_score_difference':data['export_max_score_difference']}
        protocol['candidates'].append(row)
        write_json(destination,protocol)
        print(json.dumps({'C':c,'correct':row['validation']['correct'],
                          'PLEASE':row['validation']['per_class']['PLEASE']['correct'],
                          'MY':row['validation']['per_class']['MY']['correct'],
                          'acceptance_met':row['acceptance_met']}),flush=True)
    qualified=[r for r in protocol['candidates'] if r['acceptance_met']]
    # The evaluation helper calls its unweighted cross-entropy field "loss".
    protocol['selected']=min(qualified,key=lambda r:r['validation']['loss']) if qualified else None
    protocol['status']='complete'
    write_json(destination,protocol)
    print('Selected:',protocol['selected']['checkpoint'] if qualified else 'none',flush=True)


if __name__=='__main__':main()
