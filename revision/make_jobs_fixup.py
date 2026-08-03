"""Fix-up / reinforcement jobs after the first full campaign:

A. Diffusion-reaction re-run with L=pi (paper domain; the first campaign
   inherited L=1 from main_run_R.py, under which the exact solution violates
   the enforced BCs): ff_baseline 75 + ff_supp 15 jobs.
B. Helmholtz (2,10) trials 5-9 for all 5 models (high seed-variance cell).
C. Kovasznay (2,10) robustness at N_r=10000 (main_run_LPA.py setting;
   primary comparison stays at the unified N_r=5000): exp 'ff_nr10k'.
"""
import json
import os

from make_jobs import (CONDITIONS, DIFF, HELM, KOVA, LPA_CFG, N_TRIALS,
                       SIGMAS, job)


def dr_rerun_jobs():
    jobs = []
    order, panels = LPA_CFG[DIFF]
    for nh, nn in CONDITIONS:
        for trial in range(N_TRIALS):
            jobs.append(job(DIFF, 'ff_baseline', 'R', nh, nn, trial, ['--key', 'R']))
            jobs.append(job(DIFF, 'ff_baseline', 'LPA_P%d_N%d' % (order, panels), nh, nn, trial,
                            ['--key', 'LPA', '--order', str(order), '--panels', str(panels)]))
            for s in SIGMAS:
                jobs.append(job(DIFF, 'ff_baseline', 'FF_s%g' % s, nh, nn, trial,
                                ['--key', 'FF', '--sigma', str(s)]))
    for s in SIGMAS:
        for trial in range(N_TRIALS):
            jobs.append(job(DIFF, 'ff_supp', 'FF_s%g_lr0.001' % s, 2, 10, trial,
                            ['--key', 'FF', '--sigma', str(s), '--lr', '0.001']))
    return jobs


def helm_extra_trial_jobs():
    jobs = []
    order, panels = LPA_CFG[HELM]
    for trial in range(N_TRIALS, 2 * N_TRIALS):
        jobs.append(job(HELM, 'ff_baseline', 'R', 2, 10, trial, ['--key', 'R']))
        jobs.append(job(HELM, 'ff_baseline', 'LPA_P%d_N%d' % (order, panels), 2, 10, trial,
                        ['--key', 'LPA', '--order', str(order), '--panels', str(panels)]))
        for s in SIGMAS:
            jobs.append(job(HELM, 'ff_baseline', 'FF_s%g' % s, 2, 10, trial,
                            ['--key', 'FF', '--sigma', str(s)]))
    return jobs


def kova_nr10k_jobs():
    jobs = []
    order, panels = LPA_CFG[KOVA]
    for trial in range(N_TRIALS):
        jobs.append(job(KOVA, 'ff_nr10k', 'R', 2, 10, trial, ['--key', 'R', '--nr', '10000']))
        jobs.append(job(KOVA, 'ff_nr10k', 'LPA_P%d_N%d' % (order, panels), 2, 10, trial,
                        ['--key', 'LPA', '--order', str(order), '--panels', str(panels),
                         '--nr', '10000']))
        for s in SIGMAS:
            jobs.append(job(KOVA, 'ff_nr10k', 'FF_s%g' % s, 2, 10, trial,
                            ['--key', 'FF', '--sigma', str(s), '--nr', '10000']))
    return jobs


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    jobs = dr_rerun_jobs() + helm_extra_trial_jobs() + kova_nr10k_jobs()
    with open(os.path.join(here, 'jobs_fixup.json'), 'w') as f:
        json.dump(jobs, f, indent=1)
    print('jobs_fixup.json: %d jobs' % len(jobs))
