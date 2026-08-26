"""Job list for the polynomial-order sweep at every depth (Figures 2 and 4).

The revision campaign ran the order study only at 2x10 on Helmholtz, so the
convergence panels of Figures 2 and 4 had no order curves at 3x10 and 4x10
under the unified protocol; the only per-order runs at those depths were the
original ones, taken at a different collocation budget. This fills the gap so
both figures come from a single protocol.

Existing configurations are not re-run and nothing is overwritten: the new runs
go to the experiment tag `order_sweep`, and the plotting code resolves each
(depth, order) across ff_baseline / low_order / order_sweep.

  Helmholtz : P5 at 2x10, and P in {2,3,4,5} at 3x10 and 4x10   (45 runs)
  Kovasznay : P in {2,3,4,5} at 3x10 and 4x10, N_r = 5000       (40 runs)

Seeds are 1234 + trial with trials 0..4, matching every other campaign.
"""
import json
import os

from paths import bench_dir

HELM = bench_dir('helmholtz2d')
KOVA = bench_dir('kovasznay')

EXP = 'order_sweep'
PANELS = 30
N_TRIALS = 5
KOVA_NR = 5000          # canonical budget (decision 5)

MISSING = [
    (HELM, 2, [5]),
    (HELM, 3, [2, 3, 4, 5]),
    (HELM, 4, [2, 3, 4, 5]),
    (KOVA, 3, [2, 3, 4, 5]),
    (KOVA, 4, [2, 3, 4, 5]),
]


def job(dir_, nh, order, trial):
    key_id = f'LPA_P{order}_N{PANELS}'
    stem = f'{nh}_10_{key_id}_{trial}'
    args = ['--key', 'LPA', '--order', str(order), '--panels', str(PANELS),
            '--nh', str(nh), '--nn', '10', '--trial', str(trial), '--exp', EXP]
    if dir_ == KOVA:
        args += ['--nr', str(KOVA_NR)]
    return {
        'dir': dir_,
        'script': 'revision_run.py',
        'args': args,
        'log': f'results/revision/{EXP}/log_{stem}.txt',
        'done_check': f'results/revision/{EXP}/run_{stem}.json',
    }


def main():
    jobs = []
    # depth-major so the 3x10 panels of both figures complete first
    for nh in (3, 4, 2):
        for dir_, d, orders in MISSING:
            if d != nh:
                continue
            for order in orders:
                for trial in range(N_TRIALS):
                    jobs.append(job(dir_, nh, order, trial))
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, 'jobs_order_sweep.json')
    with open(out, 'w') as f:
        json.dump(jobs, f, indent=1)
    print(f'wrote {out}: {len(jobs)} jobs')
    by = {}
    for j in jobs:
        k = (os.path.basename(j['dir']), j['args'][7], j['args'][3])
        by[k] = by.get(k, 0) + 1
    for k in sorted(by):
        print(f'  {k[0]:<20} {k[1]}x10  P{k[2]}: {by[k]} trials')


if __name__ == '__main__':
    main()
