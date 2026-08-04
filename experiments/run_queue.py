"""Simple job queue for unified-protocol experiment runs.

Reads a jobs JSON file (list of {"dir", "args", "log", "done_check"}) and runs
up to --workers subprocesses concurrently. A job whose done_check file already
exists is skipped, so the queue is resumable. Progress and exit codes are
appended to <jobs>.status.csv.

Usage:
  python run_queue.py jobs_task1.json --workers 6
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

PYTHON = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs_file')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    with open(args.jobs_file) as f:
        jobs = json.load(f)

    status_path = args.jobs_file.replace('.json', '') + '.status.csv'

    def is_done(path):
        # a run killed mid-write leaves a truncated JSON; treat it as not-done
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                json.load(f)
            return True
        except (ValueError, OSError):
            return False

    done_before = 0
    pending = []
    for i, job in enumerate(jobs):
        done_path = os.path.join(job['dir'], job['done_check'])
        if is_done(done_path):
            done_before += 1
        else:
            pending.append((i, job))
    print('[queue] %d jobs total, %d already done, %d pending, %d workers'
          % (len(jobs), done_before, len(pending), args.workers))

    env = dict(os.environ)
    env.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    env.setdefault('TF_NUM_INTEROP_THREADS', '1')
    env.setdefault('OMP_NUM_THREADS', '2')
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'

    running = {}  # popen -> (idx, job, t0)
    completed = 0
    failed = 0
    with open(status_path, 'a', newline='') as sf:
        writer = csv.writer(sf)
        queue = list(pending)
        while queue or running:
            while queue and len(running) < args.workers:
                idx, job = queue.pop(0)
                log_path = os.path.join(job['dir'], job['log'])
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                logf = open(log_path, 'w')
                script = job.get('script', 'run_experiment.py')
                p = subprocess.Popen([PYTHON, script] + job['args'],
                                     cwd=job['dir'], stdout=logf, stderr=subprocess.STDOUT,
                                     env=env)
                running[p] = (idx, job, time.time(), logf)
                print('[queue] start #%d: %s' % (idx, ' '.join(job['args'])), flush=True)
            time.sleep(2)
            for p in list(running):
                rc = p.poll()
                if rc is None:
                    continue
                idx, job, t0, logf = running.pop(p)
                logf.close()
                dt = time.time() - t0
                ok = rc == 0 and os.path.exists(os.path.join(job['dir'], job['done_check']))
                if ok:
                    completed += 1
                else:
                    failed += 1
                writer.writerow([idx, 'ok' if ok else 'FAIL(rc=%s)' % rc,
                                 '%.1f' % dt, ' '.join(job['args'])])
                sf.flush()
                print('[queue] %s #%d in %.1fs (%d done, %d failed, %d left)'
                      % ('done' if ok else 'FAILED', idx, dt, completed, failed,
                         len(queue) + len(running)), flush=True)
    print('[queue] finished: %d ok, %d failed' % (completed, failed))


if __name__ == '__main__':
    main()
