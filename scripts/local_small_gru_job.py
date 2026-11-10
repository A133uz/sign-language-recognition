"""Background controls for the queued 64-unit baseline comparison."""

import sys
from pathlib import Path

import local_training_job as job

ROOT = Path(__file__).resolve().parents[1]
job.JOB = ROOT/'.cache/small-gru'
job.STATE = job.JOB/'job.json'
job.STOP = job.JOB/'stop.request'
job.OUTPUT = ROOT/'models/asl-gru-64-baseline'

if __name__ == '__main__':
    job.main(run_command=[sys.executable, '-u', str(ROOT/'scripts/run_small_gru.py')],
             smoke_path=ROOT/'models/asl-gru-64-smoke/status.json',
             progress_path=job.JOB/'queue.json')
