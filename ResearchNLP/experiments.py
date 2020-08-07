import gc

from ResearchNLP.util_files import file_util, function_cache
from ResearchNLP.util_files.combinatorics_util import get_all_possible_matchings
from ResearchNLP.util_files.run_experiment_util import run_several_experiments_and_normalize
from ResearchNLP.z_experiments.old_experiments.additional_experiments.heuristic_functions_comparison import *
from ResearchNLP.z_experiments.old_experiments.additional_experiments.special import *
from ResearchNLP.z_experiments.main_experiments import only_small_train_balanced as smallbalanced
from ResearchNLP.z_experiments.main_experiments.only_small_train_balanced import *
from ResearchNLP.z_experiments.main_experiments.effects_of import *


def define_random_seeds(g):
    print "g:{0}".format(str(g))
    import numpy
    import random
    ra, rb = 1554 + g, 42 + g
    print ra, rb
    random.seed(ra)
    numpy.random.seed(rb)


# partial(cn.load_mtl_16_sentiment_dataset_parameters, genre=cn.mtl_options[0]),  # [1] problematic
# partial(cn.load_mtl_16_sentiment_dataset_parameters, genre=cn.mtl_options[2]),  # [1] problematic
# cn.load_uci_spam_parameters,  # crashes because weird


#######

knowledge_bases = [
    # kb_helper.load_GloVe_model,
    # kb_helper.load_WordNet_model,
    kb_helper.load_dep_word2vec_model,
    # kb_helper.load_word2vec_model,
]
data_sets = [
    # cn.load_cornell_subjectivity_parameters,
    # cn.load_cornell_sentiment_parameters,
    # cn.load_stanford_sentiment_treebank_parameters,
    # cn.load_hate_speech_parameters,
    cn.load_codementor_sentiment_analysis_parameters,
]
init_size = 10
batch_size = 20

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # tmp use CPU


def experiment_with_params((run_num, (kb_load, data_set_load))):
    def run_experiment(g=0):  # used in experiment_on_data_and_save_results()
        cn.experiment_purpose = ""
        define_random_seeds(g)
        # data_set_load(d_measure=10)
        # cn.init_pool_size = init_size

        gc.collect()
        # return test_pos_distribution()
        # return test_percent_of_positive_generated_depending_on_dist()
        # return insert_by_init_heuristic_functions_scores(SynStateUncertainty, SynStateRandom)
        # return insert_in_AL_fashion_with_heuristic_funs(SynStateTrainDataGain, SynStateTestDataGain)
        # return test_adding_random_orig_examples()
        # return run_f1_per_addition_using_search_experiment_simult()
        # return compare_kb_f1_per_addition_random_synthesis_sa_experiment()
        # return compare_f1_per_addition_using_search_sa_experiment()
        # return compare_f1_per_addition_for_each_search_algorithm()
        # return choose_best_from_global_pool()
        # return compare_f1_per_addition_for_2_heuristics(SynStateUncertainty, SynStateUncertainty)
        # return compare_f1_per_addition_for_heuristic_and_competitors(SynStateTrainDataGain)
        # return compare_pool_generation_methods_for_f1_per_addition(select_from_pool_uncertainty)
        # return compare_pool_selection_methods_for_f1_per_addition(generate_pool_curr_pool_uncertainty)
        # return compare_generated_pool_insertion_order()
        # return compare_generation_methods_pools()
        return compare_pool_generation_methods_proper_al()
        # return compare_orig_pool_insertion_order()
        # return compare_pool_enhancement_methods()
        # return effect_of_num_of_operators()
        # return effect_of_size_of_semantic_environment()
        # return effect_of_semantic_environment_on_label_switches()

    def load_expr_env():
        cn.experiment_purpose = ""
        cn.curr_experiment_params = ''
        cn.curr_experiment_params_list = list()
        if data_set_load == cn.load_cornell_subjectivity_parameters:  # this means the next one is sent subjectivity
            define_random_seeds(g=2)  # subjectivity g=2
        else:
            define_random_seeds(g=0)
            # if run_num == 0:
            #     define_random_seeds(g=4)
            # else:
            #     define_random_seeds(g=3)
        data_set_load(d_measure=10)
        cn.init_pool_size = init_size
        print "init pool size", cn.init_pool_size
        smallbalanced.examples_at_each_step = batch_size
        print "proper AL batch size", smallbalanced.examples_at_each_step
        kb_load()

        prepare_balanced_dataset(print_expert_acc=False)
        print "initial accuracy", run_classifier(cn.base_training_df, cn.validation_data_df).acc
        # with open(cn.data_name + "_core.txt", 'w') as f:
        #     f.write('\n'.join(cn.base_training_df[cn.col_names.text].values))

    # cn.load_handmade_experiments_parameters(d_measure=20)
    # kb_helper.load_dep_word2vec_model()
    # from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools import text_modops_util
    # ops = text_modops_util.synthesize_mod_operators_sent(u"Never tell a bitch what u up to..", cn.data_df, cn.col_names)

    retval = run_several_experiments_and_normalize(run_experiment, load_expr_env, run_num, run_count=5)

    experiment_name, table_headers, data, plot_fun, experiment_dir_abspath = retval

    # kb_helper.save_previous_model()
    plt.draw()
    print "total_mod_ops: " + str(cn.total_mod_ops)
    print "inst_count: " + str(cn.inst_count)
    # from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools import text_modops_util
    # print "average mod ops", float(text_modops_util.mod_op_sum) / text_modops_util.call_count
    batch_experiment_folder = "/home/yonatanz/Dropbox/Research/Interesting results/"

    try:
        if experiment_name == 'compare_pool_generation_methods_proper_AL':
            batch_experiment_folder += "balanced small train/different batch size proper AL/d_measure{0}/" \
                                       "core_set {1}/total new sents {2}/pool size {3}/rand_0.25/{4} suppress 3/".format(  # /g="+str(3+run_num)+"/
                str(cn.distance_measure), str(init_size), str(smallbalanced.total_new_sents),
                str(smallbalanced.pool_size_each_step), str(batch_size))
        elif experiment_name == 'small_train_compare_generation_methods_pools':
            batch_experiment_folder += "balanced small train/different core set size/d_measure{0}/{1}/".format(
                str(cn.distance_measure), str(init_size))
        else:
            batch_experiment_folder += "balanced small train/{0}/d_measure{1}/".format(
                experiment_name, str(cn.distance_measure))

        file_util.makedirs(batch_experiment_folder)
        plt.savefig(batch_experiment_folder + cn.data_name + ".png")
        csv_filename = "_".join([cn.data_name.replace("/", "_"), experiment_name, kb_helper.kb_type, cn.expr_id]) + ".csv"
        csv_filepath = os.path.join(experiment_dir_abspath, csv_filename)
        with open(batch_experiment_folder + "experiment_csv_files.txt", "a+") as f:
            f.write(csv_filepath + "\n")
        with open(batch_experiment_folder + "experiment_folders.txt", "a+") as f:
            f.write(experiment_dir_abspath + "\n")
    except Exception as e:
        print experiment_name
        print e
        with open("/home/yonatanz/Dropbox/Research/Interesting results/experiment_csv_files.txt", "a+") as f:
            f.write(experiment_dir_abspath + "\n")  # write experiment path if for some reason doesnt work
        print experiment_dir_abspath

    plt.show()


if __name__ == '__main__':
    if len(get_all_possible_matchings(knowledge_bases, data_sets)) != 1:
        from ResearchNLP.util_files.parallel_load import ParallelLoad
        ParallelLoad.wait_for_all_loads()
    # map(experiment_with_params, list(enumerate(get_all_possible_matchings(knowledge_bases, data_sets))))
    # parmap(experiment_with_params, list(enumerate(get_all_possible_matchings(knowledge_bases, data_sets))), nprocs=3)
    function_cache.save_cache()

    for size in [5]:  # [20, 10, 5]:
        batch_size = size
        map(experiment_with_params, list(enumerate(get_all_possible_matchings(knowledge_bases, data_sets))))
        # parmap(experiment_with_params, list(enumerate(get_all_possible_matchings(knowledge_bases, data_sets))), nprocs=3)
        function_cache.save_cache()
