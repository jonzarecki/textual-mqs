import cPickle as pkl
import csv
import os
import time

import numpy as np
from matplotlib import pyplot as plt, gridspec

from ResearchNLP.util_files import ExprScores
from ResearchNLP.util_files.combinatorics_util import lists_maxlen

fig_counter = 1
our_results_colors = ['r', 'b', 'g', 'y', "#7c1c5d", "#ffae42"]
our_results_markers = ['.', '*', 'p', 's', 'x', '+']
our_results_line_styles = ['-', '--', '-.', ':', '--', '-.']
baseline_results_colors = ['gainsboro', 'm', 'c', "#ff9933", "#996633"]
baseline_results_markers = ['v', '^', '<', '>']
baseline_results_line_styles = [':', '-.', '--', '-.']


def define_new_figure(size=(20, 5)):
    global fig_counter
    fig = plt.figure(fig_counter, figsize=size)
    fig_counter += 1
    return fig


def share_axis_between_rows(axis_mat, axis):
    # type: (np.ndarray, str) -> None
    if axis == 'x':
        axis_extract = lambda ax: ax.get_xlim()
        axis_set = lambda ax, lims: ax.set_xlim(lims)
    else:
        axis_extract = lambda ax: ax.get_ylim()
        axis_set = lambda ax, lims: ax.set_ylim(lims)

    for r in range(axis_mat.shape[0]):
        row = axis_mat[r, :]
        min_ax = min(map(lambda ax: axis_extract(ax)[0], row))
        max_ax = max(map(lambda ax: axis_extract(ax)[1], row))
        map(lambda ax: axis_set(ax, (min_ax, max_ax)), row)


def share_axis_between_rows(axis_mat, axis):
    # type: (np.ndarray, str) -> None
    if axis == 'x':
        axis_extract = lambda ax: ax.get_xlim()
        axis_set = lambda ax, lims: ax.set_xlim(lims)
    else:
        axis_extract = lambda ax: ax.get_ylim()
        axis_set = lambda ax, lims: ax.set_ylim(lims)

    for r in range(axis_mat.shape[0]):
        row = axis_mat[r, :]
        min_ax = min(map(lambda ax: axis_extract(ax)[0], row))
        max_ax = max(map(lambda ax: axis_extract(ax)[1], row))
        map(lambda ax: axis_set(ax, (min_ax, max_ax)), row)


def prepare_footnote_text():
    from ResearchNLP import Constants as cn
    from ResearchNLP.knowledge_bases import kb_helper
    txt = ""
    txt += 'Dataset: ' + cn.data_name + ", "
    txt += 'knowledge-base: ' + kb_helper.kb_type + ", "
    txt += 'Feature extractor: ' + cn.Feature_Extractor.__name__ + ", "
    txt += 'Distance measure: ' + str(cn.distance_measure) + ", "
    txt += "\n"
    txt += 'Inner pred model: ' + cn.Inner_PredictionModel.__name__ + ", "
    txt += 'Expert model: ' + cn.Expert_PredictionModel.__name__ + ", "
    txt += 'Expert feature extractor: ' + cn.Expert_FeatureExtractor.__name__ + ", "
    txt += 'Core set size: ' + str(cn.init_pool_size) + ","
    txt += "\n"
    txt += cn.expr_id + "     " + cn.experiment_purpose
    txt += "\n"
    txt += cn.curr_experiment_params
    return txt


def plot_minimum_maximum_classifier_performance(ax):
    from ResearchNLP.z_experiments import experiment_util
    max_acc = experiment_util.check_all_train_data_accuracy()
    min_acc = experiment_util.check_initial_train_data_accuracy()
    ax.axhline(max_acc, color='#000000', linestyle='dashed', linewidth=1)
    ax.axhline(min_acc, color='#000000', linestyle='dashed', linewidth=1)


def plot_experiment_multiple_metrics(x_axis_vals, our_ys, our_labels, baselines_ys, baselines_labels, title, x_axis):
    assert len(x_axis_vals) >= lists_maxlen(*(our_ys + baselines_ys)), \
        "x_axis_vals should hold the x values for all results"
    assert len(ExprScores.enumerate_score_types()) <= 3, "subplot size too small"
    gs = gridspec.GridSpec(9, 3)

    fig = define_new_figure()
    fig.suptitle(title)
    fig.text(0, 0.0, prepare_footnote_text())
    for type_num, (type_name, list_to_type) in enumerate(ExprScores.enumerate_score_types()):
        ax = fig.add_subplot(gs[0:7, type_num])

        for i, result_list in enumerate(our_ys):
            ax.plot(x_axis_vals[:len(result_list)], list_to_type(result_list), our_results_colors[i],
                    linewidth=2.5, label=our_labels[i])

        for i, baseline_list in enumerate(baselines_ys):
            ax.plot(x_axis_vals[:len(baseline_list)], list_to_type(baseline_list), baseline_results_colors[i],
                    linewidth=2.5, label=baselines_labels[i])

        ax.set_xlabel(x_axis)
        ax.set_ylabel(type_name + " score")
        ax.legend(loc='best')
        #plt.tight_layout()


def plot_experiment_single_metric(x_axis_vals, our_ys, our_labels, baselines_ys, baselines_labels, title, x_axis, y_axis,
                                  print_min_max=True):
    assert len(x_axis_vals) >= lists_maxlen(*(our_ys + baselines_ys)), \
        "x_axis_vals should hold the x values for all results"

    fig = define_new_figure(size=(10, 5))
    gs = gridspec.GridSpec(9, 1)
    ax = fig.add_subplot(gs[0:7, 0])

    for i, result_list in enumerate(our_ys):
        ax.plot(x_axis_vals[:len(result_list)], result_list, our_results_colors[i],
                linewidth=2.5, label=our_labels[i])

    for i, baseline_list in enumerate(baselines_ys):
        ax.plot(x_axis_vals[:len(baseline_list)], baseline_list, baseline_results_colors[i],
                linewidth=2.5, label=baselines_labels[i])

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.legend(loc='best')
    if print_min_max:
        plot_minimum_maximum_classifier_performance(ax)
    fig.suptitle(title)
    fig.text(0, 0.0, prepare_footnote_text())
    #plt.tight_layout()


def plot_histogram(values_our, values_our_labels, values_baselines, values_baselines_lables, title, x_axis, y_axis):
    fig = define_new_figure()
    gs = gridspec.GridSpec(9, 1)
    ax = fig.add_subplot(gs[0:7, 0])

    ax.hist(values_our + values_baselines,
            label=values_our_labels + values_baselines_lables,
            color=our_results_colors[:len(values_our)] + baseline_results_colors[:len(values_baselines)])
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.legend(loc='best')
    fig.suptitle(title)
    fig.text(0, 0.0, prepare_footnote_text())
    #plt.tight_layout()


def plot_experiment_publish_format(plt_ax, x_axis_vals, our_ys, our_labels, baselines_ys, baselines_labels,
                                   x_axis, y_axis):
    assert len(x_axis_vals) >= lists_maxlen(*(our_ys + baselines_ys)), \
        "x_axis_vals should hold the x values for all results"

    for i, result_list in enumerate(our_ys):
        plt_ax.plot(x_axis_vals[:len(result_list)], result_list, our_results_colors[i],
                    linestyle=our_results_line_styles[i], linewidth=2.5, label=our_labels[i],
                    marker=our_results_markers[i], markersize=12)

    for i, baseline_list in enumerate(baselines_ys):
        plt_ax.plot(x_axis_vals[:len(baseline_list)], baseline_list, baseline_results_colors[i],
                    linestyle=baseline_results_line_styles[i], linewidth=2.5, label=baselines_labels[i],
                    marker=baseline_results_markers[i], markersize=12)

    plt_ax.xaxis.set_tick_params(labelsize=26)
    plt_ax.yaxis.set_tick_params(labelsize=26)
    plt_ax.set_xlabel(x_axis, fontsize=26)
    plt_ax.set_ylabel(y_axis, fontsize=26)
    # plt_ax.legend(loc='best')


def save_figure_publishable(fig, f_path):
    plt.figure(fig.number)
    plt.tight_layout(pad=1.0)
    plt.draw()
    time.sleep(2)
    fig.savefig(f_path, pad_inches=1.0, dpi=400)


def extract_info_from_csv(csv_path, baseline_st_idx):
    with open(csv_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        tbl_headers = reader.next()
        x_axis = tbl_headers[0]
        our_labels = tbl_headers[1:baseline_st_idx]
        baselines_labels = tbl_headers[baseline_st_idx:]
        x_axis_vals = []
        our_ys = map(lambda _: [], range(len(our_labels)))
        baselines_ys = map(lambda _: [], range(len(baselines_labels)))
        for line in reader:
            x_axis_vals.append(int(line[0]))
            for i in range(1, baseline_st_idx):
                if line[i] != '':
                    our_ys[i - 1].append(float(line[i]))
            for i in range(baseline_st_idx, len(line)):
                if line[i] != '':
                    baselines_ys[i - baseline_st_idx].append(float(line[i]))

    return x_axis, x_axis_vals, our_ys, our_labels, baselines_ys, baselines_labels


def extract_info_from_expr_folder(fold_path):
    expr_id = os.path.basename(fold_path)
    csv_path = os.path.join(fold_path, 'avg_results'+expr_id+'.csv')
    with open(os.path.join(fold_path, 'all_exprs_data_points'+expr_id+'.pkl'), 'rb') as f:
        all_exprs_data_points = pkl.load(f)
    return csv_path, all_exprs_data_points


def extract_std_deviation_all_exprs_data_points(all_expr_data_points):
    all_expr_data_points_std = []

    # iterate over each result-list
    for res_all_data_points in all_expr_data_points:
        curr_res_data_points_std = []

        for all_expr_point in res_all_data_points:  # go through all the experiments data points
            curr_res_data_points_std.append(np.std(all_expr_point))

        all_expr_data_points_std.append(curr_res_data_points_std)

    return all_expr_data_points_std
