import os
from collections import namedtuple

import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import savgol_filter

from experiments.analysis.experiment_database import ExperimentDatabase, get_key_from_scalar_configs

sns.set_style('whitegrid')

Metric = namedtuple('Metric', 'x_label, y_label, x_log, y_log')


def plot_epoch_y_curves(metric_dict, title, xlabel, ylabel, foldername, x_log, y_log, smoothing=False,
                        filename=None, max_iter=None):
    plt.figure()
    for config, values in metric_dict.items():
        if smoothing:
            values = [savgol_filter(value, 11, 3) for value in values]

        mean, std = np.mean(values, axis=0), np.std(values, axis=0) / np.sqrt(len(values))

        if max_iter:
            mean, std = mean[:max_iter], std[:max_iter]
            plt.plot(np.arange(0, max_iter, 1), mean, label=config)
            plt.fill_between(np.arange(0, max_iter, 1), mean - std, mean + std, alpha=0.3)
        else:
            plt.plot(np.arange(len(mean)), mean, label=config)
            plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
    ax = plt.gca()
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    plt.xlim(left=0, right=len(mean))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    folder_path = os.path.join(os.getcwd(), foldername)
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, '{}.pdf'.format(filename)),
    plt.savefig(filepath[0])
    plt.close()


def plot_correlation_between_epochs(metric_dict, title, xlabel, ylabel, foldername, x_log, y_log, smoothing=False,
                                    filename=None, max_iter=None, start=0):
    config_markerstyle = ['*', 'o', '-']
    plt.figure(figsize=(6, 2))
    for metric_idx, (config, values) in enumerate(metric_dict.items()):
        for epoch_idx, (epoch, val) in enumerate(values.items()):
            if epoch_idx == 0:
                label = config
            else:
                label = None
            plt.plot(np.arange(5, 50, 10), val, label=label, marker=config_markerstyle[metric_idx],
                     color=plt.cm.magma(epoch_idx / len(values)))
    ax = plt.gca()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Create legend for the optimizer
    # plt.legend(loc=1)

    # Create legend for the fidelity
    epoch_color_path_legend = [mpatches.Patch(color=plt.cm.magma(epoch_idx / len(values)), label=epoch) for
                               epoch_idx, epoch in enumerate(['4 epochs', '12 epochs', '36 epochs', '108 epochs'])]
    # plt.legend(handles=epoch_color_path_legend, loc=2)

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    folder_path = os.path.join(os.getcwd(), foldername)
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, '{}.pdf'.format(filename)),
    plt.savefig(filepath[0])
    plt.close()


def plot_epoch_twin_y_curves(metric_dict_left, metric_dict_right, title, xlabel, ylabel_left, ylabel_right, foldername,
                             x_log, y_log, smoothing=False, filename=None, max_iter=None):
    fig, ax_left = plt.subplots()
    ax_left.set_ylabel(ylabel_left)
    for config, values in metric_dict_left.items():
        if smoothing:
            values = [savgol_filter(value, 11, 3) for value in values]

        mean, std = np.mean(values, axis=0), np.std(values, axis=0) / np.sqrt(len(values))

        if max_iter:
            mean, std = mean[:max_iter], std[:max_iter]
            ax_left.plot(np.arange(0, max_iter, 1), mean, label=config)
            ax_left.fill_between(np.arange(0, max_iter, 1), mean - std, mean + std, alpha=0.3)
        else:
            ax_left.plot(np.arange(len(mean)), mean, label=config)
            ax_left.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    ax_right = ax_left.twinx()  # instantiate a second axes that shares the same x-axis
    ax_right.set_ylabel(ylabel_right)

    for config, values in metric_dict_right.items():
        if smoothing:
            values = [savgol_filter(value, 11, 3) for value in values]

        mean, std = np.mean(values, axis=0), np.std(values, axis=0) / np.sqrt(len(values))

        if max_iter:
            mean, std = mean[:max_iter], std[:max_iter]
            ax_right.plot(np.arange(0, max_iter, 1), mean, linestyle='-.', alpha=0.4)
            ax_right.fill_between(np.arange(0, max_iter, 1), mean - std, mean + std, linestyle=':', alpha=0.1)
        else:
            ax_right.plot(np.arange(len(mean)), mean, linestyle='-.', alpha=0.4)
            ax_right.fill_between(np.arange(len(mean)), mean - std, mean + std, linestyle=':', alpha=0.1)

    ax = plt.gca()
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax_left.set_yscale('log')
        ax_right.set_yscale('log')

    plt.xlim(left=0, right=len(mean))
    plt.title(title)
    ax_left.set_xlabel(xlabel)

    plt.tight_layout()
    folder_path = os.path.join(os.getcwd(), foldername)
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, '{}.pdf'.format(filename)),
    plt.savefig(filepath[0])
    plt.close()


def plot_x_y_curve(x, y, title, xlabel, ylabel, foldername, filename=None):
    plt.figure()
    for index in range(len(x)):
        x_i, y_i = x[index, :], y[index, :]
        plt.scatter(x_i, y_i, color=[plt.cm.magma(i) for i in np.linspace(0, 1, len(x_i))])
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(bottom=min(y.flatten()))
    plt.xlim(left=min(x.flatten()))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    folder_path = os.path.join(os.getcwd(), foldername)
    filepath = os.path.join(folder_path, '{}_{}_{}.pdf'.format(filename, xlabel, ylabel))
    plt.savefig(filepath[0])
    plt.close()


def darts_weight_decay_plot(darts_experiment_database, darts_consistency_experiment_database, gdas_experiment_database,
                            random_ws_database, metric_dict, search_space):
    weight_decay_1e4 = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0001, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_3e4 = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_9e4 = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0009, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_27e4 = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0027, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_81e4 = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0081, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})

    for metric_key, metric in metric_dict.items():
        plot_epoch_y_curves(
            metric_dict={'Weight Decay 1e-4': get_key_from_scalar_configs(weight_decay_1e4, metric_key),
                         'Weight Decay 3e-4': get_key_from_scalar_configs(weight_decay_3e4, metric_key),
                         'Weight Decay 9e-4': get_key_from_scalar_configs(weight_decay_9e4, metric_key),
                         'Weight Decay 27e-4': get_key_from_scalar_configs(weight_decay_27e4,
                                                                           metric_key),
                         'Weight Decay 81e-4': get_key_from_scalar_configs(weight_decay_81e4,
                                                                           metric_key),
                         },
            ylabel=metric.y_label, xlabel=metric.x_label, title=None,
            foldername='experiments/plot_export',
            filename='weight_decay_search_space1_{}'.format(metric_key), x_log=metric.x_log, y_log=metric.y_log)

        plot_epoch_twin_y_curves(
            metric_dict_left={
                'Weight Decay 1e-4': get_key_from_scalar_configs(weight_decay_1e4, metric_key),
                'Weight Decay 3e-4': get_key_from_scalar_configs(weight_decay_3e4, metric_key),
                'Weight Decay 9e-4': get_key_from_scalar_configs(weight_decay_9e4, metric_key),
                'Weight Decay 27e-4': get_key_from_scalar_configs(weight_decay_27e4,
                                                                  metric_key),
                'Weight Decay 81e-4': get_key_from_scalar_configs(weight_decay_81e4,
                                                                  metric_key),
            },
            metric_dict_right={
                'Weight Decay 1e-4': get_key_from_scalar_configs(weight_decay_1e4, 'one_shot_validation_errors'),
                'Weight Decay 3e-4': get_key_from_scalar_configs(weight_decay_3e4, 'one_shot_validation_errors'),
                'Weight Decay 9e-4': get_key_from_scalar_configs(weight_decay_9e4, 'one_shot_validation_errors'),
                'Weight Decay 27e-4': get_key_from_scalar_configs(weight_decay_27e4,
                                                                  'one_shot_validation_errors'),
                'Weight Decay 81e-4': get_key_from_scalar_configs(weight_decay_81e4,
                                                                  'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label, ylabel_right='Validation Error (OS)',
            foldername='experiments/plot_export',
            filename='weight_decay_search_space_{}_twin_x_{}'.format(search_space, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)


def gdas_weight_decay(darts_experiment_database, darts_consistency_experiment_database, gdas_experiment_database,
                      random_ws_database, metric_dict, search_space):
    weight_decay_1e4 = gdas_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0001, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_3e4 = gdas_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_9e4 = gdas_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0009, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_27e4 = gdas_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0027, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_81e4 = gdas_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0081, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    for metric_key, metric in metric_dict.items():
        plot_epoch_y_curves(
            metric_dict={'Weight Decay 1e-4': get_key_from_scalar_configs(weight_decay_1e4, metric_key),
                         'Weight Decay 3e-4': get_key_from_scalar_configs(weight_decay_3e4, metric_key),
                         'Weight Decay 9e-4': get_key_from_scalar_configs(weight_decay_9e4, metric_key),
                         'Weight Decay 27e-4': get_key_from_scalar_configs(weight_decay_27e4,
                                                                           metric_key),
                         'Weight Decay 81e-4': get_key_from_scalar_configs(weight_decay_81e4,
                                                                           metric_key),
                         },
            ylabel=metric.y_label, xlabel=metric.x_label, title=None,
            foldername='experiments/plot_export',
            filename='gdas_weight_decay_search_space_3_{}'.format(metric_key), x_log=metric.x_log, y_log=metric.y_log)

        plot_epoch_twin_y_curves(
            metric_dict_left={
                'Weight Decay 1e-4': get_key_from_scalar_configs(weight_decay_1e4, metric_key),
                'Weight Decay 3e-4': get_key_from_scalar_configs(weight_decay_3e4, metric_key),
                'Weight Decay 9e-4': get_key_from_scalar_configs(weight_decay_9e4, metric_key),
                'Weight Decay 27e-4': get_key_from_scalar_configs(weight_decay_27e4,
                                                                  metric_key),
                'Weight Decay 81e-4': get_key_from_scalar_configs(weight_decay_81e4,
                                                                  metric_key),
            },
            metric_dict_right={
                'Weight Decay 1e-4': get_key_from_scalar_configs(weight_decay_1e4, 'one_shot_validation_errors'),
                'Weight Decay 3e-4': get_key_from_scalar_configs(weight_decay_3e4, 'one_shot_validation_errors'),
                'Weight Decay 9e-4': get_key_from_scalar_configs(weight_decay_9e4, 'one_shot_validation_errors'),
                'Weight Decay 27e-4': get_key_from_scalar_configs(weight_decay_27e4,
                                                                  'one_shot_validation_errors'),
                'Weight Decay 81e-4': get_key_from_scalar_configs(weight_decay_81e4,
                                                                  'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label, ylabel_right='Validation Error (OS)',
            foldername='experiments/plot_export',
            filename='gdas_weight_decay_search_space_{}_twin_x_{}'.format(search_space, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)


def pc_darts_weight_decay(darts_experiment_database, darts_consistency_experiment_database, gdas_experiment_database,
                          pc_darts_database, random_ws_database, metric_dict, search_space):
    weight_decay_1e4 = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0001, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_3e4 = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_9e4 = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0009, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_27e4 = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0027, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    weight_decay_81e4 = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0081, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    for metric_key, metric in metric_dict.items():
        plot_epoch_y_curves(
            metric_dict={'Weight Decay 1e-4': get_key_from_scalar_configs(weight_decay_1e4, metric_key),
                         'Weight Decay 3e-4': get_key_from_scalar_configs(weight_decay_3e4, metric_key),
                         'Weight Decay 9e-4': get_key_from_scalar_configs(weight_decay_9e4, metric_key),
                         'Weight Decay 27e-4': get_key_from_scalar_configs(weight_decay_27e4,
                                                                           metric_key),
                         'Weight Decay 81e-4': get_key_from_scalar_configs(weight_decay_81e4,
                                                                           metric_key),
                         },
            ylabel=metric.y_label, xlabel=metric.x_label, title=None,
            foldername='experiments/plot_export',
            filename='pc_darts_weight_decay_search_space_{}_{}'.format(search_space, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)

        plot_epoch_twin_y_curves(
            metric_dict_left={
                'Weight Decay 1e-4': get_key_from_scalar_configs(weight_decay_1e4, metric_key),
                'Weight Decay 3e-4': get_key_from_scalar_configs(weight_decay_3e4, metric_key),
                'Weight Decay 9e-4': get_key_from_scalar_configs(weight_decay_9e4, metric_key),
                'Weight Decay 27e-4': get_key_from_scalar_configs(weight_decay_27e4,
                                                                  metric_key),
                'Weight Decay 81e-4': get_key_from_scalar_configs(weight_decay_81e4,
                                                                  metric_key),
            },
            metric_dict_right={
                'Weight Decay 1e-4': get_key_from_scalar_configs(weight_decay_1e4, 'one_shot_validation_errors'),
                'Weight Decay 3e-4': get_key_from_scalar_configs(weight_decay_3e4, 'one_shot_validation_errors'),
                'Weight Decay 9e-4': get_key_from_scalar_configs(weight_decay_9e4, 'one_shot_validation_errors'),
                'Weight Decay 27e-4': get_key_from_scalar_configs(weight_decay_27e4,
                                                                  'one_shot_validation_errors'),
                'Weight Decay 81e-4': get_key_from_scalar_configs(weight_decay_81e4,
                                                                  'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label, ylabel_right='Validation Error (OS)',
            foldername='experiments/plot_export',
            filename='pc_darts_weight_decay_search_space_{}_twin_x_{}'.format(search_space, metric_key),
            x_log=metric.x_log,
            y_log=metric.y_log)


def pc_darts_learning_rate(darts_experiment_database, darts_consistency_experiment_database, gdas_experiment_database,
                           pc_darts_database, random_ws_database, metric_dict, search_space):
    learning_rate_0_25 = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.25})
    learning_rate_0_025 = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    learning_rate_0_0025 = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.0025})
    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'Learning Rate 0.25': get_key_from_scalar_configs(learning_rate_0_25, metric_key),
                'Learning Rate 0.025': get_key_from_scalar_configs(learning_rate_0_025, metric_key),
                'Learning Rate 0.0025': get_key_from_scalar_configs(learning_rate_0_0025, metric_key),
            },
            metric_dict_right={
                'Learning Rate 0.25': get_key_from_scalar_configs(learning_rate_0_25, 'one_shot_validation_errors'),
                'Learning Rate 0.025': get_key_from_scalar_configs(learning_rate_0_025, 'one_shot_validation_errors'),
                'Learning Rate 0.0025': get_key_from_scalar_configs(learning_rate_0_0025, 'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label, ylabel_right='Validation Error (OS)',
            foldername='experiments/plot_export',
            filename='pc_darts_learning_rate_search_space_{}_twin_x_{}'.format(search_space, metric_key),
            x_log=metric.x_log,
            y_log=metric.y_log)


def gdas_learning_rate(darts_experiment_database, darts_consistency_experiment_database, gdas_experiment_database,
                       pc_darts_database, random_ws_database, metric_dict, search_space):
    learning_rate_0_25 = gdas_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.25})
    learning_rate_0_025 = gdas_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    learning_rate_0_0025 = gdas_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.0025})
    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'Learning Rate 0.25': get_key_from_scalar_configs(learning_rate_0_25, metric_key),
                'Learning Rate 0.025': get_key_from_scalar_configs(learning_rate_0_025, metric_key),
                'Learning Rate 0.0025': get_key_from_scalar_configs(learning_rate_0_0025, metric_key),
            },
            metric_dict_right={
                'Learning Rate 0.25': get_key_from_scalar_configs(learning_rate_0_25, 'one_shot_validation_errors'),
                'Learning Rate 0.025': get_key_from_scalar_configs(learning_rate_0_025, 'one_shot_validation_errors'),
                'Learning Rate 0.0025': get_key_from_scalar_configs(learning_rate_0_0025, 'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label, ylabel_right='Validation Error (OS)',
            foldername='experiments/plot_export',
            filename='gdas_learning_rate_search_space_{}_twin_x_{}'.format(search_space, metric_key),
            x_log=metric.x_log,
            y_log=metric.y_log)


def darts_learning_rate(darts_experiment_database, darts_consistency_experiment_database, gdas_experiment_database,
                        pc_darts_database, random_ws_database, metric_dict, search_space):
    learning_rate_0_25 = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.25})
    learning_rate_0_025 = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    learning_rate_0_0025 = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'weight_decay': 0.0003, 'epochs': 100,
         'warm_start_epochs': 0, 'learning_rate': 0.0025})
    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'Learning Rate 0.25': get_key_from_scalar_configs(learning_rate_0_25, metric_key),
                'Learning Rate 0.025': get_key_from_scalar_configs(learning_rate_0_025, metric_key),
                'Learning Rate 0.0025': get_key_from_scalar_configs(learning_rate_0_0025, metric_key),
            },
            metric_dict_right={
                'Learning Rate 0.25': get_key_from_scalar_configs(learning_rate_0_25, 'one_shot_validation_errors'),
                'Learning Rate 0.025': get_key_from_scalar_configs(learning_rate_0_025, 'one_shot_validation_errors'),
                'Learning Rate 0.0025': get_key_from_scalar_configs(learning_rate_0_0025, 'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label, ylabel_right='Validation Error (OS)',
            foldername='experiments/plot_export',
            filename='darts_learning_rate_search_space_{}_twin_x_{}'.format(search_space, metric_key),
            x_log=metric.x_log,
            y_log=metric.y_log)


def darts_warm_start_plot(darts_experiment_database, darts_consistency_experiment_database, gdas_experiment_database,
                          random_ws_database, metric_dict, search_space):
    darts_first_order = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': None})
    darts_warm_start = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 20})
    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'DARTS (first order) w/o warm start': get_key_from_scalar_configs(darts_first_order, metric_key),
                'DARTS (first order) w/ warm start': get_key_from_scalar_configs(darts_warm_start, metric_key),
            },
            metric_dict_right={
                'DARTS (first order) w/o warm start': get_key_from_scalar_configs(darts_first_order,
                                                                                  'one_shot_validation_errors'),
                'DARTS (first order) w/ warm start': get_key_from_scalar_configs(darts_warm_start,
                                                                                 'one_shot_validation_errors')},
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label + ' (-)', ylabel_right='Validation Error (OS) (-.-)',
            foldername='experiments/plot_export',
            filename='darts_warm_starting_ss_{}_{}'.format(search_space, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)


def darts_warm_start_cutout_plot(darts_experiment_database, darts_consistency_experiment_database,
                                 gdas_experiment_database, random_ws_database, metric_dict, search_space):
    darts_first_order = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': None})
    darts_warm_start = darts_experiment_database.query(
        {'unrolled': False, 'cutout': True, 'search_space': search_space, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 20})
    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'DARTS (first order) w/o warm start, w/o cutout': get_key_from_scalar_configs(darts_first_order,
                                                                                              metric_key),
                'DARTS (first order) w/ warm start, w/ cutout': get_key_from_scalar_configs(darts_warm_start,
                                                                                            metric_key),
            },
            metric_dict_right={
                'DARTS (first order) w/o warm start, w/o cutout': get_key_from_scalar_configs(darts_first_order,
                                                                                              'one_shot_validation_errors'),
                'DARTS (first order) w/ warm start, w/ cutout': get_key_from_scalar_configs(darts_warm_start,
                                                                                            'one_shot_validation_errors')},
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label + ' (-)', ylabel_right='Validation Error (OS) (-.-)',
            foldername='experiments/plot_export',
            filename='darts_warm_starting_cutout_ss_{}_{}'.format(search_space, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)


def gdas_warm_start_plot(darts_experiment_database, darts_consistency_experiment_database, gdas_experiment_database,
                         random_ws_database, metric_dict, search_space):
    gdas_no_cutout = gdas_experiment_database.query(
        {'search_space': search_space, 'cutout': False, 'unrolled': False, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': None})
    gdas_warm_start = gdas_experiment_database.query(
        {'search_space': search_space, 'cutout': False, 'unrolled': False, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 20})
    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'GDAS w/o warm start': get_key_from_scalar_configs(gdas_no_cutout, metric_key),
                'GDAS w/ warm start': get_key_from_scalar_configs(gdas_warm_start, metric_key),
            },
            metric_dict_right={
                'GDAS w/o warm start': get_key_from_scalar_configs(gdas_no_cutout, 'one_shot_validation_errors'),
                'GDAS w/ warm start': get_key_from_scalar_configs(gdas_warm_start, 'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label + ' (-)', ylabel_right='Validation Error (OS) (-.-)',
            foldername='experiments/plot_export',
            filename='gdas_comp_ss_{}_warm_start{}'.format(search_space, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)


def pc_darts_warm_start_plot(darts_experiment_database, darts_consistency_experiment_database, gdas_experiment_database,
                             pc_darts_database, random_ws_database, metric_dict, search_space):
    pc_darts_no_cutout = pc_darts_database.query(
        {'search_space': search_space, 'cutout': False, 'unrolled': False, 'epochs': 50, 'weight_decay': 0.0003,
         'warm_start_epochs': 0})
    pc_darts_warm_start = pc_darts_database.query(
        {'search_space': search_space, 'cutout': True, 'unrolled': False, 'epochs': 50, 'weight_decay': 0.0003,
         'warm_start_epochs': 20})
    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'PC-DARTS w/o warm start': get_key_from_scalar_configs(pc_darts_no_cutout, metric_key),
                'PC-DARTS w/ warm start w/ cutout': get_key_from_scalar_configs(pc_darts_warm_start, metric_key),
            },
            metric_dict_right={
                'PC-DARTS w/o warm start w/o cutout': get_key_from_scalar_configs(pc_darts_no_cutout,
                                                                                  'one_shot_validation_errors'),
                'PC-DARTS w/ warm start w/ cutout': get_key_from_scalar_configs(pc_darts_warm_start,
                                                                                'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label + ' (-)', ylabel_right='Validation Error (OS) (-.-)',
            foldername='experiments/plot_export',
            filename='pc_darts_comp_ss_{}_warm_start{}'.format(search_space, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)


def pc_darts_cutout_plot(darts_experiment_database, darts_consistency_experiment_database,
                         gdas_experiment_database,
                         pc_darts_database, random_ws_database, metric_dict, search_space):
    pc_darts_no_cutout = pc_darts_database.query(
        {'search_space': search_space, 'cutout': False, 'unrolled': False, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    pc_darts_cutout = pc_darts_database.query(
        {'search_space': search_space, 'cutout': True, 'unrolled': False, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    pc_darts_second_order = pc_darts_database.query(
        {'search_space': search_space, 'cutout': False, 'unrolled': True, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    pc_darts_second_order_cutout = pc_darts_database.query(
        {'search_space': search_space, 'cutout': True, 'unrolled': True, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'PC-DARTS (first order) w/o cutout': get_key_from_scalar_configs(pc_darts_no_cutout, metric_key),
                'PC-DARTS (first order) w/ cutout': get_key_from_scalar_configs(pc_darts_cutout,
                                                                                metric_key),
                # 'PC-DARTS (second order) w/o cutout': get_key_from_scalar_configs(pc_darts_second_order, metric_key),
                # 'PC-DARTS (second order) w/ cutout': get_key_from_scalar_configs(pc_darts_second_order_cutout,
                #                                                                 metric_key),
            },
            metric_dict_right={
                'PC-DARTS (first order) w/o cutout': get_key_from_scalar_configs(pc_darts_no_cutout,
                                                                                 'one_shot_validation_errors'),
                'PC-DARTS (first order) w/ cutout': get_key_from_scalar_configs(pc_darts_cutout,
                                                                                'one_shot_validation_errors'),
                # 'PC-DARTS (second order) w/o cutout': get_key_from_scalar_configs(pc_darts_second_order,
                #                                                                  'one_shot_validation_errors'),
                # 'PC-DARTS (second order) w/ cutout': get_key_from_scalar_configs(pc_darts_second_order_cutout,
                #                                                                 'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label + ' (-)',
            ylabel_right='Validation Error (OS) (-.-)',
            foldername='experiments/plot_export',
            filename='pc_darts_comp_ss_{}_cutout_{}'.format(search_space, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)


def gdas_cutout_plot(darts_experiment_database, darts_consistency_experiment_database,
                     gdas_experiment_database,
                     pc_darts_database, random_ws_database, metric_dict, search_space):
    gdas = gdas_experiment_database.query(
        {'search_space': search_space, 'cutout': False, 'unrolled': False, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})

    gdas_cutout = gdas_experiment_database.query(
        {'search_space': search_space, 'cutout': True, 'unrolled': False, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})

    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'GDAS w/o cutout': get_key_from_scalar_configs(gdas, metric_key),
                'GDAS w/ cutout': get_key_from_scalar_configs(gdas_cutout, metric_key),
            },
            metric_dict_right={
                'GDAS w/o cutout': get_key_from_scalar_configs(gdas, 'one_shot_validation_errors'),
                'GDAS w/ cutout': get_key_from_scalar_configs(gdas_cutout, 'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label + ' (-)',
            ylabel_right='Validation Error (OS) (-.-)',
            foldername='experiments/plot_export',
            filename='gdas_comp_ss_{}_cutout_{}'.format(search_space, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)


def do_plots_for_search_space(search_space_number):
    # Correlation Plot
    darts_corr_no_cutout = darts_experiment_database.query_correlation(
        {'search_space': search_space_number, 'cutout': False, 'epochs': 50, 'learning_rate': 0.025})
    plot_correlation_between_epochs(
        metric_dict={
            'DARTS w/o cutout': darts_corr_no_cutout[0]['scalars']['correlation_total'],
        }, ylabel='Spearman Correlation', title='DARTS (first order) w/o cutout', xlabel='Epoch', x_log=False,
        y_log=False,
        foldername='experiments/plot_export', filename='correlation_ss_{}_darts_w_o_cutout'.format(search_space_number))

    pcdarts_corr_no_cutout = pc_darts_database.query_correlation(
        {'search_space': search_space_number, 'cutout': False, 'epochs': 50, 'learning_rate': 0.025})
    plot_correlation_between_epochs(
        metric_dict={
            'PC-DARTS': pcdarts_corr_no_cutout[0]['scalars']['correlation_total'],
        }, ylabel='Spearman Correlation', title='PC-DARTS', xlabel='Epoch', x_log=False, y_log=False,
        foldername='experiments/plot_export', filename='correlation_ss_{}_pc_darts'.format(search_space_number))

    gdas_corr_no_cutout = gdas_experiment_database.query_correlation(
        {'search_space': search_space_number, 'cutout': False})
    plot_correlation_between_epochs(
        metric_dict={
            'GDAS': gdas_corr_no_cutout[0]['scalars']['correlation_total'],
        }, ylabel='Spearman Correlation', title='GDAS', xlabel='Epoch', x_log=False, y_log=False,
        foldername='experiments/plot_export', filename='correlation_ss_{}_gdas'.format(search_space_number))

    random_ws = random_ws_database.query_correlation({'search_space': search_space_number, 'cutout': False})
    plot_correlation_between_epochs(
        metric_dict={
            'Random WS': random_ws[0]['scalars']['correlation_total'],
        }, ylabel='Spearman Correlation', title='Random WS', xlabel='Epoch', x_log=False, y_log=False, start=10,
        foldername='experiments/plot_export', filename='correlation_ss_{}_randomws'.format(search_space_number))

    ## CUTOUT
    pc_darts_cutout_plot(darts_experiment_database=darts_experiment_database,
                         darts_consistency_experiment_database=darts_consistency_experiment_database,
                         gdas_experiment_database=gdas_experiment_database, random_ws_database=random_ws_database,
                         metric_dict=metric_dict, pc_darts_database=pc_darts_database, search_space=search_space_number)

    gdas_cutout_plot(darts_experiment_database=darts_experiment_database,
                     darts_consistency_experiment_database=darts_consistency_experiment_database,
                     gdas_experiment_database=gdas_experiment_database, random_ws_database=random_ws_database,
                     metric_dict=metric_dict, pc_darts_database=pc_darts_database, search_space=search_space_number)

    # DARTS Cutout comparison
    darts_first_order = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space_number, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    darts_first_order_cutout = darts_experiment_database.query(
        {'unrolled': False, 'cutout': True, 'search_space': search_space_number, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    darts_second_order = darts_experiment_database.query(
        {'unrolled': True, 'cutout': False, 'search_space': search_space_number, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    darts_second_order_cutout = darts_experiment_database.query(
        {'unrolled': True, 'cutout': True, 'search_space': search_space_number, 'epochs': 100, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'DARTS (first order) w/o cutout': get_key_from_scalar_configs(darts_first_order, metric_key),
                'DARTS (first order) w/ cutout': get_key_from_scalar_configs(darts_first_order_cutout, metric_key),
                'DARTS (second order) w/o cutout': get_key_from_scalar_configs(darts_second_order, metric_key),
                'DARTS (second order) w/ cutout': get_key_from_scalar_configs(darts_second_order_cutout,
                                                                              metric_key),
            },
            metric_dict_right={
                'DARTS (first order) w/o cutout': get_key_from_scalar_configs(darts_first_order,
                                                                              'one_shot_validation_errors'),
                'DARTS (first order) w/ cutout': get_key_from_scalar_configs(darts_first_order_cutout,
                                                                             'one_shot_validation_errors'),
                'DARTS (second order) w/o cutout': get_key_from_scalar_configs(darts_second_order,
                                                                               'one_shot_validation_errors'),
                'DARTS (second order) w/ cutout': get_key_from_scalar_configs(darts_second_order_cutout,
                                                                              'one_shot_validation_errors')},
            title=None, xlabel=metric.x_label, ylabel_left=metric.y_label + ' (-)',
            ylabel_right='Validation Error (OS) (-.-)', foldername='experiments/plot_export',
            filename='second_order_vs_first_order_cutout_no_cutout_ss_{}_{}'.format(search_space_number, metric_key),
            x_log=metric.x_log, y_log=metric.y_log)

    ## WEIGHT DECAY
    darts_weight_decay_plot(darts_experiment_database=darts_experiment_database,
                            darts_consistency_experiment_database=None,
                            gdas_experiment_database=gdas_experiment_database, random_ws_database=random_ws_database,
                            metric_dict=metric_dict, search_space=search_space_number)
    gdas_weight_decay(darts_experiment_database=darts_experiment_database, darts_consistency_experiment_database=None,
                      gdas_experiment_database=gdas_experiment_database, random_ws_database=random_ws_database,
                      metric_dict=metric_dict, search_space=search_space_number)
    pc_darts_weight_decay(darts_experiment_database=darts_experiment_database,
                          darts_consistency_experiment_database=darts_consistency_experiment_database,
                          gdas_experiment_database=gdas_experiment_database, random_ws_database=random_ws_database,
                          metric_dict=metric_dict, pc_darts_database=pc_darts_database,
                          search_space=search_space_number)

    ## LEARNING RATE
    pc_darts_learning_rate(darts_experiment_database=darts_experiment_database,
                           darts_consistency_experiment_database=darts_consistency_experiment_database,
                           gdas_experiment_database=gdas_experiment_database, random_ws_database=random_ws_database,
                           metric_dict=metric_dict, pc_darts_database=pc_darts_database,
                           search_space=search_space_number)

    gdas_learning_rate(darts_experiment_database=darts_experiment_database,
                       darts_consistency_experiment_database=darts_consistency_experiment_database,
                       gdas_experiment_database=gdas_experiment_database, random_ws_database=random_ws_database,
                       metric_dict=metric_dict, pc_darts_database=pc_darts_database,
                       search_space=search_space_number)

    darts_learning_rate(darts_experiment_database=darts_experiment_database,
                        darts_consistency_experiment_database=darts_consistency_experiment_database,
                        gdas_experiment_database=gdas_experiment_database, random_ws_database=random_ws_database,
                        metric_dict=metric_dict, pc_darts_database=pc_darts_database,
                        search_space=search_space_number)

    ## OPTIMIZER COMPARISON
    darts_first_order = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space_number, 'epochs': 50, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    darts_second_order = darts_experiment_database.query(
        {'unrolled': True, 'cutout': False, 'search_space': search_space_number, 'epochs': 50, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    random_ws = random_ws_database.query({'search_space': search_space_number, 'epochs': 50})
    gdas_no_cutout = gdas_experiment_database.query(
        {'search_space': search_space_number, 'cutout': False, 'unrolled': False, 'epochs': 50, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    pc_darts = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space_number, 'epochs': 50, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    enas = enas_database.query({
        'search_space': search_space_number, 'epochs': 50
    })

    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'DARTS (first order)': get_key_from_scalar_configs(darts_first_order, metric_key),
                'DARTS (second order)': get_key_from_scalar_configs(darts_second_order, metric_key),
                'GDAS': get_key_from_scalar_configs(gdas_no_cutout, metric_key),
                'PC-DARTS': get_key_from_scalar_configs(pc_darts, metric_key),
                'ENAS': get_key_from_scalar_configs(enas, metric_key),
                'Random WS': get_key_from_scalar_configs(random_ws, metric_key),
            },
            metric_dict_right={
                'DARTS (first order)': get_key_from_scalar_configs(darts_first_order, 'one_shot_validation_errors'),
                'DARTS (second order)': get_key_from_scalar_configs(darts_second_order, 'one_shot_validation_errors'),
                'GDAS': get_key_from_scalar_configs(gdas_no_cutout, 'one_shot_validation_errors'),
                'PC-DARTS': get_key_from_scalar_configs(pc_darts, 'one_shot_validation_errors'),
                'ENAS': get_key_from_scalar_configs(enas, 'one_shot_validation_errors'),
                'Random WS': get_key_from_scalar_configs(random_ws, 'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label + ' (-)', ylabel_right='Validation Error (OS) (-.-)',
            foldername='experiments/plot_export',
            filename='optimizer_comparison_ss_{}_50_{}'.format(search_space_number, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)

    ## OPTIMIZER COMPARISON
    darts_first_order = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space_number, 'epochs': 25, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    darts_second_order = darts_experiment_database.query(
        {'unrolled': True, 'cutout': False, 'search_space': search_space_number, 'epochs': 25, 'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})

    for metric_key, metric in metric_dict.items():
        plot_epoch_y_curves(
            metric_dict={
                'DARTS (first order)': get_key_from_scalar_configs(darts_first_order, metric_key),
                'DARTS (second order)': get_key_from_scalar_configs(darts_second_order, metric_key),
            }, title=None, xlabel=metric.x_label, ylabel=metric.y_label, foldername='experiments/plot_export',
            filename='optimizer_comparison_ss_{}_25_{}'.format(search_space_number, metric_key), x_log=metric.x_log,
            y_log=metric.y_log)

    ## OPTIMIZER COMPARISON
    darts_first_order = darts_experiment_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space_number, 'epochs': 100,
         'weight_decay': 0.0003,
         'warm_start_epochs': 0, 'learning_rate': 0.025})
    darts_second_order = darts_experiment_database.query(
        {'unrolled': True, 'cutout': False, 'search_space': search_space_number, 'epochs': 100,
         'weight_decay': 0.0003, 'warm_start_epochs': 0, 'learning_rate': 0.025})
    random_ws = random_ws_database.query({'search_space': search_space_number, 'epochs': 100})
    gdas_no_cutout = gdas_experiment_database.query(
        {'search_space': search_space_number, 'cutout': False, 'unrolled': False, 'epochs': 100,
         'weight_decay': 0.0003, 'warm_start_epochs': 0, 'learning_rate': 0.025})
    pc_darts = pc_darts_database.query(
        {'unrolled': False, 'cutout': False, 'search_space': search_space_number, 'epochs': 100,
         'weight_decay': 0.0003, 'warm_start_epochs': 0, 'learning_rate': 0.025})
    enas = enas_database.query({
        'search_space': search_space_number, 'epochs': 100
    })

    for metric_key, metric in metric_dict.items():
        plot_epoch_twin_y_curves(
            metric_dict_left={
                'DARTS (first order)': get_key_from_scalar_configs(darts_first_order, metric_key),
                'DARTS (second order)': get_key_from_scalar_configs(darts_second_order, metric_key),
                'GDAS': get_key_from_scalar_configs(gdas_no_cutout, metric_key),
                'PC-DARTS': get_key_from_scalar_configs(pc_darts, metric_key),
                'ENAS': get_key_from_scalar_configs(enas, metric_key),
                'Random WS': get_key_from_scalar_configs(random_ws, metric_key),
            },
            metric_dict_right={
                'DARTS (first order)': get_key_from_scalar_configs(darts_first_order, 'one_shot_validation_errors'),
                'DARTS (second order)': get_key_from_scalar_configs(darts_second_order, 'one_shot_validation_errors'),
                'GDAS': get_key_from_scalar_configs(gdas_no_cutout, 'one_shot_validation_errors'),
                'PC-DARTS': get_key_from_scalar_configs(pc_darts, 'one_shot_validation_errors'),
                'ENAS': get_key_from_scalar_configs(enas, 'one_shot_validation_errors'),
                'Random WS': get_key_from_scalar_configs(random_ws, 'one_shot_validation_errors'),
            },
            title=None,
            xlabel=metric.x_label, ylabel_left=metric.y_label + ' (-)', ylabel_right='Validation Error (OS) (-.-)',
            foldername='experiments/plot_export',
            filename='optimizer_comparison_ss_{}_100_{}'.format(search_space_number, metric_key),
            x_log=metric.x_log,
            y_log=metric.y_log)


def main():
    do_plots_for_search_space('1')
    do_plots_for_search_space('2')
    do_plots_for_search_space('3')


if __name__ == '__main__':
    enas_database = ExperimentDatabase(root_dir=os.path.join('experiments', 'enas'))

    metric_dict = {
        'validation_errors': Metric(x_label='Epoch', y_label='Validation Regret (NB)', x_log=False, y_log=True),
        'test_errors': Metric(x_label='Epoch', y_label='Test Regret (NB)', x_log=False, y_log=True),
        'one_shot_validation_errors': Metric(x_label='Epoch', y_label='Validation Error (OS)', x_log=False,
                                             y_log=True),
        'one_shot_training_errors': Metric(x_label='Epoch', y_label='Training Error (OS)', x_log=False, y_log=True)
    }
    darts_experiment_database = ExperimentDatabase(root_dir=os.path.join('experiments', 'darts_trans'))
    darts_consistency_experiment_database = None
    gdas_experiment_database = ExperimentDatabase(root_dir=os.path.join('experiments', 'gdas_trans'))
    pc_darts_database = ExperimentDatabase(root_dir=os.path.join('experiments', 'pc_darts_trans'))
    random_ws_database = ExperimentDatabase(root_dir=os.path.join('experiments', 'random_ws'))

    main()
