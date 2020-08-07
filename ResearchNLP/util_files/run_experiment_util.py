import datetime
import os.path
import time
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pkl

from ResearchNLP import Constants as cn
from ResearchNLP.util_files import file_util, Tee, pandas_util
from ResearchNLP.util_files.file_util import list_all_files_in_folder
from ResearchNLP.util_files.multiproc_util import parmap
from config import CODE_DIR, expr_log_dir_abspath, tmp_expr_folder_prefix


def run_several_experiments_and_normalize(experiment_fun, load_expr_env, run_num, run_count=2):
    def run_averaged_experiment(expr_f):
        def mod_experiment_func(g):
            print "starting averaged run: {0}".format(g)
            start_time = time.time()
            res = expr_f(g)
            print "averaged run {0} run time: {1:.2f} minutes".format(g, (time.time() - start_time) / 60.0)
            return res

        experiments_results = parmap(lambda g: mod_experiment_func(g), range(run_count), nprocs=3)
        # experiments_results = map(lambda g: mod_experiment_func(g), range(run_count))

        experiment_name, table_headers, _, plot_fun = experiments_results[0]  # average results
        avg_data_points = []
        all_data_points = []
        x_axis = experiments_results[0][2][0]
        experiments_results = map(lambda expr_idx: experiments_results[expr_idx][2][1:], range(run_count))  # no x-axis

        # iterate over each result-list
        for res_list_idx in range(len(experiments_results[0])):
            curr_res_avg_data_points = []
            curr_res_all_data_points = []

            all_expr_curr_res_dp = map(lambda expr_res: expr_res[res_list_idx], experiments_results)
            # use the minimum item count for each data point lists
            curr_res_item_count = min(map(len, all_expr_curr_res_dp))

            for itm_idx in range(curr_res_item_count):  # go through all data points for this result-list
                all_itm_results = map(lambda expr_res: expr_res[itm_idx], all_expr_curr_res_dp)
                curr_res_avg_data_points.append(np.average(all_itm_results))
                curr_res_all_data_points.append(all_itm_results)

            avg_data_points.append(curr_res_avg_data_points)
            all_data_points.append(curr_res_all_data_points)

        # import pdb
        # pdb.set_trace()
        for i in range(len(all_data_points)):
            print all_data_points[i][-1]

        cn.experiment_purpose = "averaged " + str(run_count) + "runs, " + cn.experiment_purpose
        experiment_data = {'avg_data_points': [x_axis] + avg_data_points,  # add x_axis
                           'all_data_points': all_data_points,
                           'x_axis': x_axis}
        return experiment_name, table_headers, experiment_data, plot_fun

    return experiment_on_data_and_save_results(lambda: run_averaged_experiment(experiment_fun), load_expr_env, run_num)


def experiment_on_data_and_save_results(experiment_fun, load_expr_env, run_num):
    """
    runs the given experiment function with loaded configurations in Constants,
        saves the results as .csv and figure .png
    :param experiment_fun: the wanted experiment function
    :return: None
    """
    start_time = time.time()
    expr_id = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')  # used in all files to identify this experiment
    cn.expr_id = expr_id
    experiment_dir_curr_relpath = os.path.join(cn.curr_experiment_params, expr_id)

    file_util.makedirs(tmp_expr_folder_prefix, exists_ok=True)
    cn.tmp_expr_foldpath = file_util.create_temp_folder(prefix=tmp_expr_folder_prefix)
    python_files_in_dir = list_all_files_in_folder(CODE_DIR, "py", recursively=True)
    file_util.copy_files_while_keeping_structure(python_files_in_dir, CODE_DIR, cn.tmp_expr_foldpath)

    load_expr_env()  # load after copying files

    with Tee(os.path.join(cn.tmp_expr_foldpath, 'output_log'+expr_id+'.txt')):
        print "run number: " + str(run_num)
        print experiment_dir_curr_relpath  # not the actual relpath
        experiment_name, table_headers, expr_results, plot_fun = experiment_fun()

        if type(expr_results) == dict:
            data_points = expr_results['avg_data_points']
            all_exprs_data_points = expr_results['all_data_points']
            with open(os.path.join(cn.tmp_expr_foldpath, 'all_exprs_data_points'+expr_id+'.pkl'), 'wb') as f:
                pkl.dump(all_exprs_data_points, f)
        else:
            data_points = expr_results

        pandas_util.save_lists_as_csv(table_headers, data_points, os.path.join(cn.tmp_expr_foldpath,
                                                                               'avg_results'+expr_id+'.csv'))

        if all_exprs_data_points in plot_fun.__code__.co_varnames:
            plot_fun(experiment_name, table_headers, data_points, all_exprs_data_points=all_exprs_data_points)
        else:
            plot_fun(experiment_name, table_headers, data_points)
        plt.savefig(os.path.join(cn.tmp_expr_foldpath, 'figure_avg_results'+expr_id+'.png'), trnsparent=True, dpi=500)
        print "total run time: {0:.2f} minutes".format((time.time() - start_time) / 60.0)

    # experiment finished !
    experiment_dir_abspath = os.path.join(expr_log_dir_abspath, experiment_name, cn.curr_experiment_params, expr_id)
    print experiment_dir_abspath

    os.makedirs(experiment_dir_abspath)  # copy temp folder to the destination folder
    file_util.copy_folder_contents(src_dir=cn.tmp_expr_foldpath, dst_dir=experiment_dir_abspath)
    file_util.delete_folder_with_content(cn.tmp_expr_foldpath)

    return experiment_name, table_headers, data_points, plot_fun, experiment_dir_abspath
