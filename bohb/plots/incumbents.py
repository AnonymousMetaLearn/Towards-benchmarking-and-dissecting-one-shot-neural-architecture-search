import os
import argparse
import pandas as pd
import numpy as np
#from IPython import embed

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt

import result as hpres
from util import Architecture, Model, get_trajectories, get_incumbent, plot_losses, merge_and_fill_trajectories


parser = argparse.ArgumentParser(description='fANOVA analysis')
parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--working_directory', type=str, help='directory where to'
                    ' store the live rundata', default='../bohb_output')
parser.add_argument('--space', type=int, default=1, help='NASBench space')
parser.add_argument('--seed', type=int, default=1, help='Seed')
args = parser.parse_args()

s1_min = 0.05448716878890991
s2_min = 0.057592153549194336
s3_min = 0.05338543653488159


def trajectory_plot(all_runs):
    fig, ax = plt.subplots(1, figsize=(8, 4.5))

    # darts- bohb results
    for x in all_runs:
        x.time_stamps['finished'] += x.info['runtime'][0]*60

    all_runs = sorted(all_runs, key=lambda x: x.time_stamps['finished'])

    darts_results = [[x.info['test_error'][0] - eval('s{}_min'.format(args.space)),
                      x.time_stamps['finished']-x.time_stamps['started']] for x
                     in all_runs]
    darts_results = np.array(darts_results)
    cumulated_runtimes = [np.sum(darts_results[:, 1][0:i+1]) for i in
                          range(len(darts_results[:, 1]))]

    time_stamps, test_regret = get_incumbent(darts_results[:, 0],
                                             cumulated_runtimes)
    df = pd.DataFrame({'a': test_regret}, index=time_stamps)
    darts_traj = {'BOHB-DARTS': {
        'time_stamps': np.array(df.index),
        'losses': np.array(df.T)
    }}
    plot_losses(fig, ax, darts_traj, regret=False,
                plot_mean=True)

    # RE and RS
    re_results = get_trajectories(args, eval('s{}_min'.format(args.space)),
                                  path='../../experiments/discrete_optimizers',
                                  methods=['RE', 'RS', 'RL', 'SMAC', 'HB',
                                           'BOHB', 'TPE'])
    plot_losses(fig, ax, re_results, regret=False, plot_mean=True)


    ax.set_xscale('log')
    #ax.set_xlim([5e4, 5e5]) #s1
    ax.set_yscale('log')
    #ax.set_ylim([0.002, 2e-1]) #s3

    ax.set_ylabel('test regret')
    ax.set_xlabel('simulated wallclock time [s]')
    plt.legend(fontsize=10)
    plt.title("Space %d"%(args.space))
    plt.grid(True, which="both",ls="-")
    plt.tight_layout()

    os.makedirs('./incumbents', exist_ok=True)
    fig_name = './incumbents'+'/s%d-run%d-seed%d.png'%(
        args.space, args.run_id, args.seed
    )
    plt.savefig(fig_name)
    plt.show()


if __name__=='__main__':
    bohb_logs_dir = '{}/search_space_{}/run{}-seed{}'.format(
        args.working_directory, args.space, args.run_id, args.seed
    )
    res = hpres.logged_results_to_HB_result(bohb_logs_dir)


    runs = list(filter(lambda r: not (r.info is None or r.loss is None),
                       res.get_all_runs()))

    trajectory_plot(runs)
