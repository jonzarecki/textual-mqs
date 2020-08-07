from ResearchNLP.text_synthesis.heuristic_functions import find_heuristic
from ResearchNLP.text_synthesis.heuristic_functions.choose_best_for_addition import *
from ResearchNLP.util_files.matplotlib_utils.generic import plot_experiment_single_metric
from ResearchNLP.z_experiments.experiment_util import run_classifier, prepare_balanced_dataset, prepare_trn_ds
from ResearchNLP.z_experiments.ex_insertion_order import insert_in_AL_fashion, insert_in_batch_AL
from ResearchNLP.z_experiments.pool_gen_methods.generate_pool_methods import local_search_gen_template, \
    generate_pool_using_random_synthesis
from ResearchNLP import Constants as cn


def effect_of_num_of_operators():
    experiment_name = 'effect_of_num_of_operators'
    print experiment_name

    # prepare the different splits of $data_df
    balanced_train_df, validation_data_df = prepare_balanced_dataset(print_expert_acc=False)
    all_sents = list(balanced_train_df[cn.col_names.text])
    tns = 50

    final_scoring_fun = partial(lambda en_df: run_classifier(en_df, validation_data_df).acc)
    insertion_order_heuristic = find_heuristic("random-score")()
    print cn.experiment_purpose
    table_headers = ['num of operators']
    data = [[1] + range(2, 12, 2)]

    for i, heur in enumerate(["uncertainty lc LogReg", "random-score"]):  # "uncertainty lc LogReg",
        table_headers.append(heur)
        data.append(list())
        prepare_pools_funcs = list()
        prepare_pools_funcs.append(local_search_gen_template(heur, 1))
        for j in range(2, 12, 2):
            prepare_pools_funcs.append(local_search_gen_template(heur, j))

        for pool_name, prepare_pool_fun in prepare_pools_funcs:
            gen_pool_df, labeled_pool_df = prepare_pool_fun(all_sents, balanced_train_df, cn.col_names, tns)
            trn_ds, lbr, extractor = prepare_trn_ds(balanced_train_df, gen_pool_df, labeled_pool_df)
            print pool_name
            query_num, pool_insr_scores = insert_in_batch_AL(trn_ds, final_scoring_fun, lbr,
                                                             insertion_order_heuristic,
                                                             labeled_pool_df, batch_num=len(labeled_pool_df))
            data[1 + i].append(pool_insr_scores[-1])

    return experiment_name, table_headers, data, plot_effect_of_num_of_operators


def plot_effect_of_num_of_operators(_, tbl_headers, data):
    plot_experiment_single_metric(data[0], data[1:], tbl_headers[1:],
                                  [], [],
                                  'effects of number of operators on accuracy',
                                  'number of operators', "accuracy", print_min_max=False)


def effect_of_size_of_semantic_environment():
    experiment_name = 'effect_of_size_of_semantic_environment'
    print experiment_name

    # prepare the different splits of $data_df
    balanced_train_df, validation_data_df = prepare_balanced_dataset(print_expert_acc=False)
    all_sents = list(balanced_train_df[cn.col_names.text])
    tns = 50

    prepare_pools_funcs = list()
    prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 5))
    prepare_pools_funcs.append(local_search_gen_template("random-score", 5))
    prepare_pools_funcs.append(generate_pool_using_random_synthesis())
    final_scoring_fun = partial(lambda en_df: run_classifier(en_df, validation_data_df).acc)
    insertion_order_heuristic = find_heuristic("random-score")()
    print cn.experiment_purpose
    table_headers = ['size of semantic environment']
    data = [[]]

    for env_size in range(1, 5, 1) + range(5, 35, 5):
        data[0].append(env_size)
        for i, (pool_name, prepare_pool_fun) in enumerate(prepare_pools_funcs):
            if pool_name not in table_headers:
                table_headers.append(pool_name)
                data.append([])
                assert len(data) == i + 2, "meaning i+1 is our index"
            print pool_name
            cn.distance_measure = env_size
            gen_pool_df, labeled_pool_df = prepare_pool_fun(all_sents, balanced_train_df, cn.col_names, tns)
            trn_ds, lbr, extractor = prepare_trn_ds(balanced_train_df, gen_pool_df, labeled_pool_df)
            query_num, pool_insr_scores = insert_in_AL_fashion(trn_ds, final_scoring_fun, lbr,
                                                               insertion_order_heuristic,
                                                               labeled_pool_df, quota=tns)
            data[i + 1].append(pool_insr_scores[-1])

    return experiment_name, table_headers, data, plot_effect_of_size_of_semantic_environment


def plot_effect_of_size_of_semantic_environment(_, tbl_headers, data):
    plot_experiment_single_metric(data[0], data[1:], tbl_headers[1:],
                                  [], [],
                                  'effects of semantic environment size on accuracy',
                                  'semantic environment size', "accuracy", print_min_max=False)


def effect_of_semantic_environment_on_label_switches():
    experiment_name = 'effect_of_semantic_environment_on_label_switches'
    print experiment_name

    # prepare the different splits of $data_df
    balanced_train_df, validation_data_df = prepare_balanced_dataset(print_expert_acc=False)
    all_sents = list(balanced_train_df[cn.col_names.text])
    tns = 50

    prepare_pools_funcs = list()
    prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 5))
    prepare_pools_funcs.append(local_search_gen_template("random-score", 5))
    prepare_pools_funcs.append(generate_pool_using_random_synthesis())
    print cn.experiment_purpose
    table_headers = ['size of semantic environment']
    data = [[]]

    def get_num_of_label_switches(labeled_pool_df):
        count = 0
        orig_tags = dict(map(lambda (i, r): (r[cn.col_names.text], r[cn.col_names.tag]), balanced_train_df.iterrows()))
        for i, r in labeled_pool_df.iterrows():
            if orig_tags[r[cn.col_names.prev_states][-1]] != r[cn.col_names.tag]:
                count += 1
        return count

    for env_size in [10]:
        data[0].append(env_size)
        data[0].append(env_size * 2)
        for i, (pool_name, prepare_pool_fun) in enumerate(prepare_pools_funcs):
            if pool_name not in table_headers:
                table_headers.append(pool_name)
                data.append([])
                assert len(data) == i + 2, "meaning i+1 is our index"
                print pool_name
            cn.distance_measure = env_size
            gen_pool_df, labeled_pool_df = prepare_pool_fun(all_sents, balanced_train_df, cn.col_names, tns)
            data[i + 1].append(get_num_of_label_switches(labeled_pool_df))
            data[i + 1].append(get_num_of_label_switches(labeled_pool_df))
    return experiment_name, table_headers, data, plot_effect_of_semantic_environment_on_label_switches


def plot_effect_of_semantic_environment_on_label_switches(_, tbl_headers, data):
    plot_experiment_single_metric(data[0], data[1:], tbl_headers[1:],
                                  [], [],
                                  'effects of semantic environment size on label switches',
                                  'semantic environment size', "#label switches", print_min_max=False)
