"""Generate job lists for the MLST revision experiment campaign.

Experiments (exp tag -> reviewer task):
  ff_baseline  : Task 1  Fourier-feature baseline vs vanilla vs LPA (all benchmarks)
  low_order    : Task 2  LPA order P in {1,2,3,4,6} on 2D Helmholtz (P=6 reused from ff_baseline)
  dof_fixed    : Task 3  fixed (P+1)*N_panel = 210 on 2D Helmholtz ((6,30) reused)
  sensitivity  : Task 4  P x N_panel grid + lr sweep on 2D Helmholtz ((6,30)@1e-2 reused)

Run trials 0..N_TRIALS-1 (seed = 1234 + trial).
"""
import json
import os

from paths import bench_dir

HELM = bench_dir('helmholtz2d')
DIFF = bench_dir('diffusion_reaction')
KOVA = bench_dir('kovasznay')

N_TRIALS = 5
SIGMAS = [1, 5, 10]
CONDITIONS = [(2, 10), (3, 10), (4, 10)]

# paper-default LPA config per benchmark
LPA_CFG = {HELM: (6, 30), DIFF: (3, 30), KOVA: (6, 30)}


def job(dir_, exp, key_id, nh, nn, trial, extra_args):
    stem = '%s_%s_%s_%s' % (nh, nn, key_id, trial)
    return {
        'dir': dir_,
        'args': ['--nh', str(nh), '--nn', str(nn), '--trial', str(trial),
                 '--exp', exp] + extra_args,
        'log': 'results/revision/%s/log_%s.txt' % (exp, stem),
        'done_check': 'results/revision/%s/run_%s.json' % (exp, stem),
    }


def task1_jobs():
    jobs = []
    # phase A: compact config (2,10) first, all benchmarks, so headline numbers land early
    for nh, nn in CONDITIONS:
        for bench in [HELM, DIFF, KOVA]:
            order, panels = LPA_CFG[bench]
            for trial in range(N_TRIALS):
                jobs.append(job(bench, 'ff_baseline', 'R', nh, nn, trial, ['--key', 'R']))
                jobs.append(job(bench, 'ff_baseline', 'LPA_P%d_N%d' % (order, panels), nh, nn, trial,
                                ['--key', 'LPA', '--order', str(order), '--panels', str(panels)]))
                for s in SIGMAS:
                    jobs.append(job(bench, 'ff_baseline', 'FF_s%g' % s, nh, nn, trial,
                                    ['--key', 'FF', '--sigma', str(s)]))
    return jobs


def task1b_jobs():
    """Supplementary: FF-PINN with Adam warm-up lr 1e-3 (best-shot check that the
    FF stall under the default protocol is not an artifact of the aggressive lr)."""
    jobs = []
    for bench in [HELM, DIFF, KOVA]:
        for s in SIGMAS:
            for trial in range(N_TRIALS):
                jobs.append(job(bench, 'ff_supp', 'FF_s%g_lr0.001' % s, 2, 10, trial,
                                ['--key', 'FF', '--sigma', str(s), '--lr', '0.001']))
    return jobs


def task2_jobs():
    jobs = []
    for P in [1, 2, 3, 4]:  # P=6 comes from ff_baseline LPA (2,10) runs
        for trial in range(N_TRIALS):
            jobs.append(job(HELM, 'low_order', 'LPA_P%d_N%d' % (P, 30), 2, 10, trial,
                            ['--key', 'LPA', '--order', str(P), '--panels', '30']))
    return jobs


def task3_jobs():
    combos = [(1, 105), (2, 70), (3, 52), (4, 42), (5, 35)]  # (6,30) reused
    jobs = []
    for P, N in combos:
        for trial in range(N_TRIALS):
            jobs.append(job(HELM, 'dof_fixed', 'LPA_P%d_N%d' % (P, N), 2, 10, trial,
                            ['--key', 'LPA', '--order', str(P), '--panels', str(N)]))
    return jobs


def task4_jobs():
    jobs = []
    for P in [2, 3, 4, 5, 6]:
        for N in [10, 20, 30, 40]:
            if (P, N) == (6, 30):
                continue  # reused from ff_baseline
            for trial in range(N_TRIALS):
                jobs.append(job(HELM, 'sensitivity', 'LPA_P%d_N%d' % (P, N), 2, 10, trial,
                                ['--key', 'LPA', '--order', str(P), '--panels', str(N)]))
    for lr in ['0.001', '0.003', '0.03']:  # 0.01 comes from the grid
        for trial in range(N_TRIALS):
            jobs.append(job(HELM, 'sensitivity', 'LPA_P6_N30_lr%g' % float(lr), 2, 10, trial,
                            ['--key', 'LPA', '--order', '6', '--panels', '30', '--lr', lr]))
    return jobs


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    all_jobs = {
        'jobs_task1.json': task1_jobs(),
        'jobs_task1b.json': task1b_jobs(),
        'jobs_task2.json': task2_jobs(),
        'jobs_task3.json': task3_jobs(),
        'jobs_task4.json': task4_jobs(),
    }
    total = 0
    for name, jobs in all_jobs.items():
        with open(os.path.join(here, name), 'w') as f:
            json.dump(jobs, f, indent=1)
        print('%s: %d jobs' % (name, len(jobs)))
        total += len(jobs)
    combined = [j for jobs in all_jobs.values() for j in jobs]
    with open(os.path.join(here, 'jobs_all.json'), 'w') as f:
        json.dump(combined, f, indent=1)
    print('total: %d jobs (jobs_all.json)' % total)
