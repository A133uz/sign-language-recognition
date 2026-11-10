"""Independent background ST-GCN job controls; preserve all GRU runs."""

import sys
from pathlib import Path

import local_training_job as job

ROOT = Path(__file__).resolve().parents[1]
job.JOB = ROOT/'.cache/stgcn-training'
job.STATE = job.JOB/'job.json'
job.STOP = job.JOB/'stop.request'
job.OUTPUT = ROOT/'models/asl-stgcn27-v1'

if __name__ == '__main__':
    job.main(run_command=[sys.executable, '-u', str(ROOT/'scripts/run_stgcn.py')],
             smoke_path=ROOT/'models/asl-stgcn27-smoke/status.json',
             progress_path=job.JOB/'pipeline.json')
