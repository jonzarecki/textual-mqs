import pandas as pd
from pandas import DataFrame

from ResearchNLP import Constants as cn
from ResearchNLP.text_synthesis.gen_amounts import pool_size
from ResearchNLP.text_synthesis.heuristic_functions.choose_best_for_addition import choose_best_by_uncertainty
from ResearchNLP.text_synthesis.sentence_generation import _generic_synthesis_from_sents
from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools.text_modops_util import synthesize_tree_depth1_bulk
from ResearchNLP.text_synthesis.synthesis_algorithms.lecun_augmentation import LecunAugmentation
from ResearchNLP.util_files import prepare_df_columns, ColumnNames
from ResearchNLP.util_files.column_names import add_sentences_and_histories_to_df
from ResearchNLP.util_files.printing_util import print_progress


def _build_new_sents_df(sent_pool, col_names, base_sents, do_difference=True):
    # type: (set, ColumnNames, list, bool) -> pd.DataFrame
    """
        Builds a new DataFrame object to put all new sents from $sent_pool (without sents from $base_sents)
    """
    if do_difference: sent_pool = sent_pool.difference(set(base_sents))  # only new sentences

    # put all new unlabeled generated sentences in a new DataFrame
    new_sent_df = pd.DataFrame(columns=list(col_names))
    new_sent_df[col_names.text] = list(sent_pool)  # fill the text column with the new sentences
    new_sent_df = prepare_df_columns(new_sent_df, col_names)
    return new_sent_df


def curr_pool_synthesis_from_sents(base_sents, base_training_df, col_names, total_new_sents=None,
                                   choose_bulk_method=choose_best_by_uncertainty):
    # type: (list, DataFrame, ColumnNames, int, callable) -> DataFrame
    """
    generates new examples based on $base_sents using a generic algorithms
    :param base_sents: list of base sents, from which we generate new ones
    :param base_training_df: DataFrame containing all labeled sentences
    :param col_names: contains the names of the columns in the output DataFrame
    :param total_new_sents: indicates the number of sentences we want to synthesize
    :param choose_bulk_method: a method for choosing sentences to be sent for generation
    :return: an unlabeled DataFrame with the new sentences generated
    """
    print "start curr pool search map"
    total_new_sents = pool_size(total_new_sents, base_sents)
    wanted_new_sents = int(total_new_sents * 4)
    choose_amount = wanted_new_sents / 8 + 1
    cn.add_experiment_param("choose_amount_"+str(choose_amount))
    if "choose_amount" not in cn.experiment_purpose:
        cn.experiment_purpose += "curr_pool choose_amount="+str(choose_amount)+", "

    from ResearchNLP.knowledge_bases import kb_helper
    kb_helper.k_base.load_knowledgebase()  # explicitly load to help processes share memory
    # print kb_helper.kb_type

    didnt_advance_count = 0
    sent_pool = set(base_sents)
    current_pool = list(base_sents)
    sent_pool_df = base_training_df.copy(deep=True)
    print "total new sentences: " + str(wanted_new_sents)
    while len(sent_pool) - len(base_sents) <= wanted_new_sents:  # gen quarter size
        all_new_tuples = synthesize_tree_depth1_bulk(current_pool, sent_pool_df, col_names)
        combined_df = add_sentences_and_histories_to_df(base_training_df, col_names, all_new_tuples)
        combined_df = prepare_df_columns(combined_df, col_names)
        chosen_idxs = choose_bulk_method(combined_df, col_names, choose_amount, sent_pool)
        if len(chosen_idxs) == 0:
            didnt_advance_count += 1

        for idx in chosen_idxs:
            sent_pool_df.loc[len(sent_pool_df)] = combined_df.loc[idx].copy(deep=True)
            # add new example to sent pools
            new_sent = combined_df[col_names.text][idx]
            assert new_sent not in sent_pool, "the new sentence should not appear beforehand"
            current_pool.append(new_sent)
            sent_pool.add(new_sent)
            print_progress(len(sent_pool) - len(base_sents), total=wanted_new_sents)
            didnt_advance_count = 0

        sent_pool_df = prepare_df_columns(sent_pool_df, col_names)

        if didnt_advance_count >= 50:
            print "didn't advance, stopping synthesis"
            break

    # use the already filled sent_pool_df
    final_chosen_idxs = choose_bulk_method(sent_pool_df, col_names, total_new_sents, set())
    new_sents_df = sent_pool_df.iloc[final_chosen_idxs].reset_index(drop=True)
    print "\ngenerated", len(new_sents_df), "sentences"
    return new_sents_df


def generate_lecun_augmentation_from_sents(base_sents, base_training_df, col_names, total_new_sents=None):
    # type: (DataFrame, ColumnNames, bool, int) -> DataFrame
    """
    generates new examples based on $base_sents using Lecun's augmentation algorithm
    :param base_sents: list of base sents, from which we generate new ones
    :param base_training_df: DataFrame of sentences, from which we create new ones
    :param col_names: contains the names of the columns in the output DataFrame
    :param total_new_sents: the amount of sentences we want to generate
    :return: an unlabeled DataFrame with the new sentences generated
    """
    print "start lecun map"
    total_new_sents = pool_size(total_new_sents, base_sents)
    lecun_alg = LecunAugmentation()
    return _generic_synthesis_from_sents(base_sents, base_training_df, col_names, lecun_alg, total_new_sents)