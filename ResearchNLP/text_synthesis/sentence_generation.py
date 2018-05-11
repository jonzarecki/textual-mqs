# coding=utf-8
import pandas as pd
from pandas import DataFrame

from ResearchNLP import Constants as cn
from ResearchNLP.text_synthesis.gen_amounts import pool_size
from ResearchNLP.text_synthesis.heuristic_functions.choose_best_for_addition import choose_best_by_uncertainty, \
    _choose_best_by_heuristic_fun
from ResearchNLP.text_synthesis.heuristic_functions.choose_best_for_expansion import choose_random_sents_from_df, \
    choose_best_using_leave_out_uncertainty
from ResearchNLP.text_synthesis.synthesis_algorithms import LocalSearchAlgorithm, RandomAlgorithm, SynthesisAlgorithm
from ResearchNLP.util_files import prepare_df_columns, ColumnNames
from ResearchNLP.util_files.column_names import add_sentences_and_histories_to_df
from ResearchNLP.util_files.printing_util import print_progress


def generate_sents_using_random_synthesis(base_sents, base_training_df, col_names, total_new_sents=None):
    # type: (DataFrame, ColumnNames, int, int) -> DataFrame
    """
    generates new examples based on $base_sents using the random synthesis algorithm
    :param base_sents: list of base sents, from which we generate new ones
    :param base_training_df: DataFrame of sentences, from which we create new ones
    :param col_names: contains the names of the columns in the output DataFrame
    :param total_new_sents: the amount of sentences we want to generate
    :return: an unlabeled DataFrame with the new sentences generated
    """
    print "start random map"
    total_new_sents = pool_size(total_new_sents, base_sents)
    rand_alg = RandomAlgorithm()
    return _generic_synthesis_from_sents(base_sents, base_training_df, col_names, rand_alg, total_new_sents)


def generate_sents_using_search_alg(base_sents, base_training_df, col_names, search_alg_code, iter_lim,
                                    total_new_sents=None):
    # type: (list, DataFrame, ColumnNames, str, int, int) -> DataFrame
    """
    generates new examples based on $base_sents using search algorithms
    :param base_sents: list of base sents, from which we generate new ones
    :param base_training_df: DataFrame of sentences, from which we create new ones
    :param col_names: contains the names of the columns in the output DataFrame
    :param search_alg_code: the algorithm code for the local search algorithm
    :param iter_lim: the limit of iterations for the search algorithm
    :param total_new_sents: the amount of sentences we want to generate
    :return: an unlabeled DataFrame with the new sentences generated
    """
    print "start search map -", cn.ss_type.__name__
    total_new_sents = pool_size(total_new_sents, base_sents)
    search_alg = LocalSearchAlgorithm(search_alg_code, iter_lim)
    sent_pool_df = _generic_synthesis_from_sents(base_sents, base_training_df, col_names, search_alg,
                                                 total_new_sents * 3, start_with_orig=False,
                                                 sents_choice_for_generation=choose_random_sents_from_df)

    # use the already filled sent_pool_df
    enhanced_df = pd.concat([base_training_df, sent_pool_df], ignore_index=True)
    final_chosen_idxs = _choose_best_by_heuristic_fun(enhanced_df, col_names, total_new_sents, cn.ss_type, set())
    new_sents_df = enhanced_df.iloc[final_chosen_idxs].reset_index(drop=True)
    print "\ngenerated", len(new_sents_df), "sentences"
    return new_sents_df


def generate_sents_using_enhanced_search(base_sents, base_training_df, col_names, search_alg_code, iter_lim, total_new_sents=None):
    # type: (list, DataFrame, ColumnNames, str, int, int) -> DataFrame
    """
    generates new examples based on $base_sents using search algorithms
    :param base_sents: list of base sents, from which we generate new ones
    :param base_training_df: DataFrame of sentences, from which we create new ones
    :param col_names: contains the names of the columns in the output DataFrame
    :param search_alg_code: the algorithm code for the local search algorithm
    :param iter_lim: the limit of iterations for the search algorithm
    :param total_new_sents: the amount of sentences we want to generate
    :return: an unlabeled DataFrame with the new sentences generated
    """
    print "start enhanced search map"
    total_new_sents = pool_size(total_new_sents, base_sents)
    search_alg = LocalSearchAlgorithm(search_alg_code, iter_lim)

    sent_pool_df = _generic_synthesis_from_sents(base_sents, base_training_df, col_names, search_alg,
                                                 total_new_sents * 2, batch_size=len(base_sents),
                                                 sents_choice_for_generation=choose_best_using_leave_out_uncertainty)

    # use the already filled sent_pool_df
    enhanced_df = pd.concat([base_training_df, sent_pool_df], ignore_index=True)
    final_chosen_idxs = choose_best_by_uncertainty(enhanced_df, col_names, total_new_sents, set())
    new_sents_df = enhanced_df.iloc[final_chosen_idxs].reset_index(drop=True)
    print "\ngenerated", len(new_sents_df), "sentences"
    return new_sents_df


def _generic_synthesis_from_sents(base_sents, base_training_df, col_names, syn_alg, total_new_sents,
                                  sents_choice_for_generation=choose_random_sents_from_df, batch_size=1,
                                  start_with_orig=False):
    # type: (list, DataFrame, ColumnNames, int, SynthesisAlgorithm, int) -> DataFrame
    """
    generates new examples based on $base_sents using a generic algorithms
    :param base_sents: list of base sents, from which we generate new ones
    :param base_training_df: DataFrame containing all labeled sentences
    :param col_names: contains the names of the columns in the output DataFrame
    :param syn_alg: a synthesis algorithm which makes the generation
    :param total_new_sents: indicates the number of sentences we want to synthesize
    :return: an unlabeled DataFrame with the new sentences generated
    """

    from ResearchNLP.knowledge_bases import kb_helper
    kb_helper.k_base.load_knowledgebase()  # explicitly load to help processes share memory
    # print kb_helper.kb_type

    didnt_advance_count = 0
    from ResearchNLP.text_synthesis.heuristic_functions import choose_best_for_expansion
    choose_best_for_expansion.counter = 0
    replicate_count = 0
    sent_pool = set(base_sents).union(set(base_training_df[col_names.text]))
    orig_sent_pool_len = len(sent_pool)
    sent_pool_df = base_training_df.copy(deep=True)
    print "total new sentences: " + str(total_new_sents)
    while len(sent_pool) - orig_sent_pool_len <= total_new_sents:  # gen quarter size
        if start_with_orig:
            chosen_sents = sents_choice_for_generation(base_training_df, col_names, batch_size)
        else:
            chosen_sents = sents_choice_for_generation(sent_pool_df, col_names, batch_size)
        new_sent_tuples = syn_alg.run_alg_parallel(sent_pool_df, col_names, chosen_sents)

        for new_sent, sent_history in new_sent_tuples:
            if new_sent not in sent_pool:
                didnt_advance_count = 0
                print_progress(len(sent_pool) - orig_sent_pool_len, total=total_new_sents)
                print str(replicate_count) + " replicated sents",
                sent_pool.add(new_sent)
                sent_pool_df = add_sentences_and_histories_to_df(sent_pool_df, col_names, [(new_sent, sent_history)])
            else:
                didnt_advance_count += 1
                replicate_count += 1

        sent_pool_df = prepare_df_columns(sent_pool_df, col_names)

        if didnt_advance_count >= 50:
            print "didn't advance, stopping synthesis"
            break

    # use the already filled sent_pool_df
    new_sents_df = sent_pool_df.iloc[len(base_training_df):(len(base_training_df)+total_new_sents)] \
        .reset_index(drop=True)
    print "generated", len(new_sents_df), "sentences"
    return new_sents_df
