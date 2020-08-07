import csv
import matplotlib
import os
import time

import numpy as np
from matplotlib import gridspec, pyplot as plt

from ResearchNLP.util_files import file_util
from generic import define_new_figure, plot_experiment_publish_format, \
    share_axis_between_rows, save_figure_publishable, extract_info_from_csv, extract_info_from_expr_folder, \
    extract_std_deviation_all_exprs_data_points


matplotlib.rcParams.update({'font.size': 12})

def plot_all_csvs_from_file(fold_paths_file, y_axis):

    fig = define_new_figure(size=(15, 14))
    fig.set_facecolor("#FFFFFF")

    gs = gridspec.GridSpec(3 * 2, 4 * 3)
    axis_list = []
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i * 2:(i + 1) * 2, j * 2 * 3:(j + 1) * 2 * 3])
            axis_list.append(ax)
    ax = fig.add_subplot(gs[4:6, 0:6])
    axis_list.append(ax)

    # fig = define_new_figure(size=(20, 8))
    # fig.set_facecolor("#FFFFFF")
    #
    # gs = gridspec.GridSpec(2, 2 * 3)
    # axis_list = []
    # for i in range(3):
    #     for j in range(1):
    #         ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
    #         axis_list.append(ax)
    # ax = fig.add_subplot(gs[1, 0:2])
    # axis_list.append(ax)
    # ax = fig.add_subplot(gs[1, 2:4])
    # axis_list.append(ax)

    titles = ('CMR', 'SUBJ', 'SST', 'HS', 'KS')
    fold_paths_list = file_util.readlines(fold_paths_file)
    for f_num, f_path in enumerate(fold_paths_list):
        csv_path, _ = extract_info_from_expr_folder(f_path)
        x_ax, x_ax_vals, our_ys, our_lbls, baselines_ys, baselines_lbls = extract_info_from_csv(csv_path, 4)
        our_lbls = ['US-BS-MQ', 'US-HC-MQ', 'S-MQ']
        baselines_lbls = ['IDEAL-RAND', 'WNA', 'RNN-LM']
        axis_list[f_num].set_title(titles[f_num], fontsize=32)
        plot_experiment_publish_format(axis_list[f_num], x_ax_vals, our_ys, our_lbls,
                                       baselines_ys[:-1], baselines_lbls[:-1], x_ax, y_axis)

    #plt.tight_layout(pad=0.3)
    plt.legend(bbox_to_anchor=(1.3, 0.95), loc=2, prop={'size': 25})
    save_figure_publishable(fig, os.path.dirname(fold_paths_file) + "/all_plots_publish.png")  # eps the format conferences want
    # for some reason the showed plot is different from the one saved
    plt.show()


def plot_all_final_results_from_file_with_error_bars(fold_paths_file):
    us_hc_scores = []
    us_hc_stds = []
    us_beam_scores = []
    us_beam_stds = []
    rand_syn_scores = []
    rand_syn_stds = []
    wna_scores = []
    wna_stds = []

    ind = np.arange(5)  # the x locations for the groups
    width = 0.20  # the width of the bars

    fig = define_new_figure(size=(10, 4.5))
    fig.set_facecolor("#FFFFFF")
    ax = fig.add_subplot(111)

    exprs_fold_paths = file_util.readlines(fold_paths_file)
    for f_num, fold_path in enumerate(exprs_fold_paths):
        csv_path, all_exprs_data_points = extract_info_from_expr_folder(fold_path)
        x_axis, x_axis_vals, our_ys, our_labels, baselines_ys, baselines_labels = extract_info_from_csv(csv_path, 4)
        ideal_scr = baselines_ys[0][-1]
        all_exprs_data_points = map(
            lambda all_res: map(lambda data_p_lst: [x / ideal_scr for x in data_p_lst], all_res),
            all_exprs_data_points)
        all_expr_data_points_std = extract_std_deviation_all_exprs_data_points(all_exprs_data_points)

        us_beam_scores.append(our_ys[0][-1] / ideal_scr)
        us_beam_stds.append(all_expr_data_points_std[0][-1] / 2)
        us_hc_scores.append(our_ys[1][-1] / ideal_scr)
        us_hc_stds.append(all_expr_data_points_std[1][-1] / 2)
        rand_syn_scores.append(our_ys[2][-1] / ideal_scr)
        rand_syn_stds.append(all_expr_data_points_std[2][-1] / 2)
        wna_scores.append(baselines_ys[1][-1] / ideal_scr)
        wna_stds.append(all_expr_data_points_std[4][-1] / 2)

        # init_score = baselines_ys[0][0]
        # all_exprs_data_points = map(lambda all_res: map(lambda data_p_lst: [(x-init_score)/(ideal_scr-init_score) for x in data_p_lst], all_res),
        #                             all_exprs_data_points)
        # all_expr_data_points_std = extract_std_deviation_all_exprs_data_points(all_exprs_data_points)
        #
        #
        # us_beam_scores.append((our_ys[0][-1]-init_score)/(ideal_scr-init_score))
        # us_beam_stds.append(all_expr_data_points_std[0][-1] / 2)
        # us_hc_scores.append((our_ys[1][-1]-init_score)/(ideal_scr-init_score))
        # us_hc_stds.append(all_expr_data_points_std[1][-1] / 2)
        # rand_syn_scores.append((our_ys[2][-1]-init_score)/(ideal_scr-init_score))
        # rand_syn_stds.append(all_expr_data_points_std[2][-1] / 2)
        # wna_scores.append((baselines_ys[1][-1]-init_score)/(ideal_scr-init_score))
        # wna_stds.append(all_expr_data_points_std[4][-1] / 2)
    ax.set_xticks(ind)

    ax.bar(ind - 1.5 * width, us_hc_scores, width, yerr=us_hc_stds,
           color='SkyBlue', label='US-HC-MQ', hatch="-")
    ax.bar(ind - 0.5 * width, us_beam_scores, width, yerr=us_beam_stds,
           color='IndianRed', label='US-BS-MQ', hatch="\\\\")
    ax.bar(ind + 0.5 * width, rand_syn_scores, width, yerr=rand_syn_stds,
           color='gold', label='S-MQ', hatch="//")
    ax.bar(ind + 1.5 * width, wna_scores, width, yerr=wna_stds,
           color='purple', label='WNA')
    # minlim, maxlim = ax.get_xlim()
    # ax.plot([minlim, maxlim], [0., 0.], "k--")
    # ax.set_xlim([minlim, maxlim])

    ax.set_ylim(0.5, 1.05)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('IDEAL accuracy ratio')
    ax.set_xticklabels(('CMR', 'SUBJ', 'SST', 'HS', 'KS'))
    #plt.tight_layout()
    ax.legend(loc='best')
    plt.show()

    save_figure_publishable(fig, os.path.dirname(fold_paths_file) + "/final_results_error_bars.png")


# orig_p = '/home/yonatanz/Dropbox/Research/Interesting results/balanced small train/different batch size proper AL/' \
#          'd_measure10/core_set 10/total new sents 100/pool size 20/rand_0.25 good/'
# csvs = map(lambda i: orig_p + str(i) + "/experiment_csv_files.txt", [5, 10, 20])
# import ResearchNLP.util_files.matplotlib_utils.tmq_specific as graph
# graph_plot_util.plot_all_csvs_from_multiple_files(csvs, 'hello', 3, 'accuracy')

# graph.plot_all_csvs_from_file(orig_p + "5/experiment_csv_files.txt", 4, 'accuracy')


def plot_all_csvs_from_multiple_files(csv_paths_files_list, title, baseline_st_idx, y_axis):
    assert len(csv_paths_files_list) == 3, "should have 3 files for 3 bulk experiments"

    fig = define_new_figure(size=(20, 25))
    fig.set_facecolor("#FFFFFF")

    gs = gridspec.GridSpec(5, 3)

    axis_mat = np.empty([5, 3], dtype=object)
    for j in range(3):
        for i in range(5):
            ax = fig.add_subplot(gs[i, j])
            axis_mat[i, j] = ax
    fig.suptitle(title)

    for j, bulk_expr_csv_f in enumerate(csv_paths_files_list):
        csv_file_paths_list = file_util.readlines(bulk_expr_csv_f)
        for f_num, f_path in enumerate(csv_file_paths_list):
            x_ax, x_ax_vals, our_ys, our_lbls, baselines_ys, baselines_lbls = extract_info_from_csv(f_path,
                                                                                                    baseline_st_idx)
            plot_experiment_publish_format(axis_mat[f_num, j], x_ax_vals, our_ys, our_lbls,
                                           baselines_ys, baselines_lbls, x_ax, y_axis)

    share_axis_between_rows(axis_mat, axis='y')
    axis_mat[0, 0].legend(loc='best')
    #plt.tight_layout()
    plt.draw()
    time.sleep(2)
    plt.show()


def plot_all_csvs_from_multiple_files_split_diff_datasets(csv_paths_files_list, baseline_st_idx, y_axis):
    assert len(csv_paths_files_list) == 3, "should have 3 files for 3 bulk experiments"

    gs = gridspec.GridSpec(1, 3)

    axis_mat = np.empty([5, 3], dtype=object)
    fig_arr = []
    for i in range(5):
        fig = define_new_figure(size=(20, 5))
        fig.set_facecolor("#FFFFFF")

        fig_arr.append(fig)
        for j in range(3):
            ax = fig.add_subplot(gs[0, j])
            axis_mat[i, j] = ax

    for j, bulk_expr_csv_f in enumerate(csv_paths_files_list):
        csv_file_paths_list = file_util.readlines(bulk_expr_csv_f)
        for f_num, f_path in enumerate(csv_file_paths_list):
            x_ax, x_ax_vals, our_ys, our_lbls, baselines_ys, baselines_lbls = extract_info_from_csv(f_path,
                                                                                                    baseline_st_idx)
            plot_experiment_publish_format(axis_mat[f_num, j], x_ax_vals, our_ys, our_lbls,
                                           baselines_ys, baselines_lbls, x_ax, y_axis)

    share_axis_between_rows(axis_mat, axis='y')
    #plt.tight_layout()

    csv_file_paths_list = file_util.readlines(csv_paths_files_list[0])
    datasets_names = map(lambda p: os.path.splitext(os.path.basename(p))[0].split('_')[0], csv_file_paths_list)

    for i, fig in enumerate(fig_arr):
        p = os.path.dirname(os.path.dirname(csv_paths_files_list[0])) + "/" + datasets_names[i] + ".pdf"
        print p
        save_figure_publishable(fig, p)
    plt.clf()


def plot_experiment_effect_of_num_of_operators_publishable(expr_fold_path):
    import csv
    fig = define_new_figure(size=(10, 5))
    fig.set_facecolor("#FFFFFF")
    ax = fig.add_subplot(111)

    csv_f_path, all_exprs_data_points = extract_info_from_expr_folder(expr_fold_path)
    with open(csv_f_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        tbl_headers = reader.next()
        x_axis = tbl_headers[0]
        our_labels = [['']]
        baselines_labels = []
        x_axis_vals = []
        our_ys = [[]]
        baselines_ys = []
        for line in reader:
            x_axis_vals.append(int(line[0]))
            our_ys[0].append(float(line[2]))

        plot_experiment_publish_format(ax, x_axis_vals, our_ys, our_labels,
                                       baselines_ys, baselines_labels, x_axis, 'final accuracy')
        # print list(csv.reader(open(csv_f_path, 'r'), delimiter='\t'))[1:]

    baselines_ys = []

    plot_experiment_publish_format(ax, x_axis_vals, our_ys, our_labels,
                                   baselines_ys, baselines_labels, x_axis, "final accuracy")
    #plt.tight_layout()

    # ax.set_ylim(0.55, 0.70)
    save_figure_publishable(fig, os.path.dirname(csv_f_path) + "/op_num_effect.pdf")
    plt.show()


def plot_label_switch_difference_publishable(fold_paths_file):
    us_hc_scores = []
    us_hc_stds = []
    rand_hc_scores = []
    rand_hc_stds = []
    rand_syn_scores = []
    rand_syn_stds = []

    ind = np.arange(5)  # the x locations for the groups
    width = 0.25  # the width of the bars
    inst_gen_c = 50  # instance gen count

    fig = define_new_figure(size=(10, 3.5))
    fig.set_facecolor("#FFFFFF")
    ax = fig.add_subplot(111)

    exprs_fold_paths = file_util.readlines(fold_paths_file)
    for f_num, fold_path in enumerate(exprs_fold_paths):
        csv_path, all_exprs_data_points = extract_info_from_expr_folder(fold_path)
        all_exprs_data_points = map(
            lambda all_res: map(lambda data_p_lst: [float(x) / inst_gen_c for x in data_p_lst], all_res),
            all_exprs_data_points)
        all_expr_data_points_std = extract_std_deviation_all_exprs_data_points(all_exprs_data_points)
        with open(csv_path, 'rb') as f:
            csv_lines = list(csv.reader(f, delimiter='\t'))
            cells = map(lambda a: float(a) / inst_gen_c, csv_lines[1])  # line with semantic env 10
            us_hc_scores.append(cells[1])
            us_hc_stds.append(all_expr_data_points_std[0][1] / 2)
            rand_hc_scores.append(cells[2])
            rand_hc_stds.append(all_expr_data_points_std[1][1] / 2)
            rand_syn_scores.append(cells[3])
            rand_syn_stds.append(all_expr_data_points_std[2][1] / 2)

    rects1 = ax.bar(ind - width, us_hc_scores, width, hatch='-',  # yerr=us_hc_stds,
                    color='SkyBlue', label='US-HC-MQ')
    rects2 = ax.bar(ind, rand_hc_scores, width, hatch='\\\\',  # yerr=rand_hc_stds,
                    color='IndianRed', label='S-HC-MQ')
    rects3 = ax.bar(ind + width, rand_syn_scores, width, hatch='//',  # yerr=rand_syn_stds,
                    color='gold', label='S-MQ')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% of switched instances')
    ax.set_xticks(ind)
    ax.set_xticklabels(('CMR', 'SUBJ', 'SST', 'HS', 'KS'))
    #plt.tight_layout()
    ax.legend(bbox_to_anchor=(0.22, 1.0), loc='upper left')

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.53, 'left': 0.47}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                    '{0:.2f}'.format(height), ha=ha[xpos], va='bottom')

    autolabel(rects1, "left")
    autolabel(rects2, "center")
    autolabel(rects3, "right")

    # print os.path.dirname(fold_paths_file) + "/label_switch.pdf"
    plt.show()
    save_figure_publishable(fig, os.path.dirname(fold_paths_file) + "/label_switch.png")


if __name__ == '__main__':
    orig_p = '/media/yonatanz/yz/Dropbox/Dropbox/Research_old/Interesting results/balanced small train/different batch size proper AL/d_measure10/core_set 10/total new sents 100/pool size 20/rand_0.25/5 OK (copy)/'
    csvs = map(lambda i: orig_p + str(i) + "/experiment_csv_files.txt", [5, 10, 20])
    plot_all_csvs_from_file(orig_p + "experiment_folders.txt", 'Accuracy')