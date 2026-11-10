"""Background cross-validation controls; baseline outputs remain separate."""

import sys
from pathlib import Path

import local_training_job as job

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'models/asl-gru-128-cv-regularized-v1'
job.JOB = ROOT/'.cache/cross-validation'
job.STATE = job.JOB/'job.json'
job.STOP = job.JOB/'stop.request'
job.OUTPUT = OUTPUT

if __name__ == '__main__':
    job.main(run_command=[sys.executable, '-u', str(ROOT/'scripts/cross_validate.py'), 'run'],
             smoke_path=OUTPUT/'smoke/status.json')
