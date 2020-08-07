import time

from ResearchNLP.text_synthesis.sentence_generation import generate_sents_using_random_synthesis
from ResearchNLP.text_synthesis.heuristic_functions import *
from ResearchNLP.text_synthesis.heuristic_functions.choose_best_for_addition import *
from ResearchNLP.util_files.matplotlib_utils.generic import plot_experiment_single_metric
from ResearchNLP.util_files.printing_util import print_progress
from ResearchNLP.z_experiments.experiment_util import run_classifier, prepare_balanced_dataset, prepare_trn_ds
from ResearchNLP.z_experiments.ex_insertion_order import insert_in_batch_AL, insert_in_AL_fashion
from ResearchNLP.z_experiments.pool_gen_methods.generate_pool_methods import prepare_pools_template, \
    curr_pool_gen_template, local_search_gen_template, prepare_orig_examples_pools, generate_pool_lecun_augmentation, \
    generate_pool_using_random_synthesis, generate_sents_using_lstm_generator
from ResearchNLP import Constants as cn
from ResearchNLP.z_experiments.pool_gen_methods.choose_from_pool_methods import select_from_pool_uncertainty


def compare_generated_pool_insertion_order():
    experiment_name = 'gen_pool_insertion_order_balanced'
    print experiment_name

    # prepare the different splits of $data_df
    balanced_train_df, validation_data_df = prepare_balanced_dataset()
    all_sents = list(balanced_train_df[cn.col_names.text])
    tns = 50

    pool_name, prep_pools = prepare_pools_template("random synthesis", generate_sents_using_random_synthesis)
    # pool_name, prep_pools = local_search_gen_template("EGL-EGL")

    generated_pool_df, labeled_pool_df = prep_pools(all_sents, balanced_train_df, cn.col_names, tns+50)
    cn.experiment_purpose += pool_name + " "

    trn_ds, lbr, extractor = prepare_trn_ds(balanced_train_df, generated_pool_df, labeled_pool_df)
    final_scoring_fun = partial(lambda en_df: run_classifier(en_df, validation_data_df).acc)

    table_headers = ['#added examples']
    data = [[0]]
    compared_heuristics = combined_heuristics_list[:] + [#("test-data-gain", lambda: SynStateTestDataGain),
                                                         ("random", lambda: "random")]

    for (heuristic_name, prepare_usage) in compared_heuristics:
        init_score = final_scoring_fun(balanced_train_df)
        ss_type = prepare_usage()
        print heuristic_name
        # query_num, heur_scores = insert_in_AL_fashion(trn_ds, final_scoring_fun, lbr, ss_type,
        #                                                    labeled_pool_df, quota=tns)
        query_num, heur_scores = insert_in_batch_AL(trn_ds, final_scoring_fun,
                                                    lbr, ss_type, labeled_pool_df, quota=tns, batch_num=5)
        heur_scores[0] = init_score
        data[0] = query_num if len(data[0]) < len(query_num) else data[0]

        data.append(heur_scores)
        table_headers.append(heuristic_name)

    # assert_ends_and_beginnings_are_the_same(data[1:])
    return experiment_name, table_headers, data, plot_compare_insertion_order


def plot_compare_insertion_order(_, table_headers, data):
    plot_experiment_single_metric(data[0], data[1:-2], table_headers[1:-2],
                                  [data[-1]] + [data[-2]], [table_headers[-1]] + [table_headers[-2]],
                                 'order of insertion using the balanced configuration', 'no. of examples added', "acc")


def compare_generation_methods_pools():
    experiment_name = 'small_train_compare_generation_methods_pools'
    print experiment_name

    # prepare the different splits of $data_df
    balanced_train_df, validation_data_df = prepare_balanced_dataset(print_expert_acc=False)
    all_sents = list(balanced_train_df[cn.col_names.text])
    tns = 40

    prepare_pools_funcs = list()
    # prepare_pools_funcs.append(curr_pool_gen_template("uncertainty lc LogReg"))
    #
    # # prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 2))
    # prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 2))
    # # prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 10))
    # # prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 0, use_enhanced=True))
    # # prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 5, use_enhanced=True))
    # # prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 10, use_enhanced=True))
    # prepare_pools_funcs.append(local_search_gen_template("random-score", 2))
    # # prepare_pools_funcs.append(local_search_gen_template("random-score", 0))
    #
    # # prepare_pools_funcs.append(curr_pool_gen_template("test-data-gain"))
    # prepare_pools_funcs.append(generate_pool_using_random_synthesis())
    # prepare_pools_funcs.append(prepare_orig_examples_pools())
    # prepare_pools_funcs.append(generate_pool_lecun_augmentation())
    prepare_pools_funcs.append(generate_sents_using_lstm_generator())

    final_scoring_fun = partial(lambda en_df: run_classifier(en_df, validation_data_df).acc)
    insertion_order_heuristic = find_heuristic("uncertainty lc LogReg")()
    # insertion_order_heuristic = find_heuristic("test-data-gain")()
    cn.add_experiment_param(insertion_order_heuristic.__name__)
    cn.experiment_purpose += "insertion order using " + insertion_order_heuristic.__name__ + " "
    print cn.experiment_purpose
    table_headers = ['#added examples']
    data = [[0]]

    for pool_name, prepare_pool_fun in prepare_pools_funcs:
        init_score = final_scoring_fun(balanced_train_df)
        print pool_name
        gen_pool_df, labeled_pool_df = prepare_pool_fun(all_sents, balanced_train_df, cn.col_names, tns)
        trn_ds, lbr, extractor = prepare_trn_ds(balanced_train_df, gen_pool_df, labeled_pool_df)
        print pool_name
        query_num, pool_insr_scores = insert_in_AL_fashion(trn_ds, final_scoring_fun, lbr,
                                                           insertion_order_heuristic,
                                                           labeled_pool_df, quota=tns)
        # query_num, pool_insr_scores = insert_in_batch_AL(trn_ds, final_scoring_fun, lbr, insertion_order_heuristic,
        #                                                  labeled_pool_df, quota=tns, batch_num=5)
        pool_insr_scores[0] = init_score
        data[0] = query_num if len(data[0]) < len(query_num) else data[0]

        table_headers.append(pool_name)
        data.append(pool_insr_scores)

    return experiment_name, table_headers, data, plot_compare_generation_methods


def plot_compare_generation_methods(_, table_headers, data):
    plot_experiment_single_metric(data[0], data[1:-2], table_headers[1:-2],
                                  [data[-2]] + [data[-1]], [table_headers[-2]] + [table_headers[-1]],
                                 'different ways of generating pools using the balanced configuration',
                                 'no. of examples added', "acc")


examples_at_each_step = 5
total_new_sents = 100
pool_size_each_step = 20


def compare_pool_generation_methods_proper_al():
    experiment_name = 'compare_pool_generation_methods_proper_AL'; print experiment_name

    # prepare the different splits of $data_df
    balanced_train_df, validation_data_df = prepare_balanced_dataset()
    all_sents = list(balanced_train_df[cn.col_names.text])

    prepare_pools_funcs = list()
    # prepare_pools_funcs.append(curr_pool_gen_template("uncertainty lc LogReg"))
    # prepare_pools_funcs.append(curr_pool_gen_template("random-score"))
    # prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 5))
    # prepare_pools_funcs.append(local_search_gen_template("uncertainty lc LogReg", 5, use_enhanced=True))
    # prepare_pools_funcs.append(local_search_gen_template("random-score", 5))
    # prepare_pools_funcs.append(generate_pool_using_random_synthesis())

    # prepare_pools_funcs.append(prepare_orig_examples_pools())
    # prepare_pools_funcs.append(generate_pool_lecun_augmentation())
    prepare_pools_funcs.append(generate_sents_using_lstm_generator())


    final_scoring_fun = partial(lambda en_df: run_classifier(en_df, validation_data_df).acc)

    cn.add_experiment_param("tns_" + str(total_new_sents))
    cn.add_experiment_param("pool_size" + str(pool_size_each_step))
    cn.add_experiment_param("batch_size" + str(examples_at_each_step))
    print cn.experiment_purpose

    def do_one_AL_cycle(pool_gen_fun, curr_training_df):
        done_generating = False
        sent_pool = set(curr_training_df[cn.col_names.text])
        gen_pool_df, labeled_pool_df = pool_gen_fun(list(sent_pool), curr_training_df, cn.col_names, pool_size_each_step)
        if len(gen_pool_df) > examples_at_each_step:
            selected_instance_idxs = select_from_pool_uncertainty(gen_pool_df, balanced_train_df, cn.col_names,
                                                                  sent_pool, examples_at_each_step)
            labeled_instances_df = labeled_pool_df.iloc[selected_instance_idxs].copy(deep=True).reset_index(drop=True)
        else:
            labeled_instances_df = labeled_pool_df  # all there is, close enough.
            if len(gen_pool_df) < examples_at_each_step:
                done_generating = True

        enriched_train_df = pd.concat([curr_training_df, labeled_instances_df], ignore_index=True)
        return enriched_train_df, final_scoring_fun(enriched_train_df), done_generating

    table_headers = ['#added examples']
    data = [range(0, total_new_sents + examples_at_each_step, examples_at_each_step)]

    for pool_name, prepare_pool_fun in prepare_pools_funcs:
        start_time = time.time()
        print "starting {0} - {1}".format(pool_name, cn.data_name)

        curr_training_df = balanced_train_df.copy(deep=True)
        res_list = [final_scoring_fun(curr_training_df)]
        for i in range(0, total_new_sents, examples_at_each_step):  # has to be serial
            print_progress(i, total=total_new_sents)

            sa = time.time()
            curr_training_df, curr_add_res, done = do_one_AL_cycle(prepare_pool_fun, curr_training_df)
            if done:
                break
            res_list.append(curr_add_res)
            print "AL cycle took {0:.2f} s".format(time.time() - sa)

        print "{0} run time: {1:.2f} minutes - {2}".format(pool_name, (time.time() - start_time) / 60.0, cn.data_name)

        table_headers.append(pool_name)
        data.append(res_list)

    return experiment_name, table_headers, data, plot_compare_pool_generation_methods_proper_al


def plot_compare_pool_generation_methods_proper_al(_, table_headers, data):
    plot_experiment_single_metric(data[0], data[1:-2], table_headers[1:-2],
                                  [data[-2]] + [data[-1]], [table_headers[-2]] + [table_headers[-1]],
                                 'compare pool generation methods - proper AL fashion',
                                 'no. of examples added', "acc")
