import copy
import random

import numpy as np
import pandas as pd
from libact.base.interfaces import Labeler
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling

from ResearchNLP import Constants as cn
from ResearchNLP.text_synthesis.heuristic_functions import *
from ResearchNLP.text_synthesis.heuristic_functions.heuristic_fun_helper import calculate_heuristic_bulk
from ResearchNLP.util_files.libact_utils import TextDataset, IdealTextLabeler
from ResearchNLP.z_experiments.experiment_util import run_classifier, run_active_learning


def insert_in_AL_fashion(trn_ds, final_score, lbr, ss_type, labeled_df=None, quota=5000, return_ins_order=False):
    # type: (TextDataset, callable, Labeler, SynState, int, bool) -> (list, list)
    E_out = []
    insertion_order = []
    trn_ds = copy.deepcopy(trn_ds)
    query_num = np.arange(0, min(quota, len(trn_ds.get_unlabeled_entries())) + 1).tolist()
    E_out.append(final_score(trn_ds.extract_labeled_dataframe()))
    for i in range(quota):
        if len(trn_ds.get_unlabeled_entries()) == 0:
            break  # finished labeling all examples
        df = trn_ds.extract_dataframe()
        cn.curr_train_df = trn_ds.extract_labeled_dataframe()
        unlabeled_idxs = list(df[df[cn.tag_col].isnull()].index)
        if ss_type == "random":  # faster than SynStateRandom
            ask_id, score = random.choice(unlabeled_idxs), -1
        else:
            idx_score_list = calculate_heuristic_bulk(df, cn.col_names, ss_type,
                                                      unlabeled_idxs, labeled_df)
            ask_id, score = max(idx_score_list, key=lambda a: a[1])

        lb = lbr.label(trn_ds.extract_sentence(ask_id))
        # print("{0}  scr: {2:.3f} lb: {3} \n{1}".format(i, trn_ds.extract_sentence(ask_id), score, lb))
        trn_ds.update(ask_id, lb)

        insertion_order.append(ask_id)
        E_out.append(final_score(trn_ds.extract_labeled_dataframe()))
        cn.last_E_out = E_out[-1]

    if return_ins_order:
        return query_num, E_out, insertion_order
    return query_num, E_out


def insert_in_initial_heuristic_order(trn_ds, final_score, lbr, ss_type, labeled_df=None, quota=5000, return_ins_order=False):
    # type: (TextDataset, callable, Labeler, SynState, int, bool) -> (list, list)
    E_out = []
    insertion_order = []
    trn_ds = copy.deepcopy(trn_ds)
    query_num = [0]
    E_out.append(final_score(trn_ds.extract_labeled_dataframe()))
    cn.last_E_out = E_out[-1]
    df = trn_ds.extract_dataframe()
    cn.curr_train_df = trn_ds.extract_labeled_dataframe()
    unlabeled_idxs_orig = list(df[df[cn.tag_col].isnull()].index)
    if ss_type == "random":
        idx_score_list = map(lambda idx: (idx, random.random()), unlabeled_idxs_orig)
    else:
        idx_score_list = calculate_heuristic_bulk(df, cn.col_names, ss_type, unlabeled_idxs_orig, labeled_df)

    for i in range(1, quota+1):
        if len(trn_ds.get_unlabeled_entries()) == 0:
            break  # finished labeling all examples
        cn.curr_train_df = trn_ds.extract_labeled_dataframe()
        ask_id, score = max(idx_score_list, key=lambda a: a[1])
        idx_score_list.remove((ask_id, score))

        lb = lbr.label(trn_ds.extract_sentence(ask_id))
        # print("{0}  scr: {2:.3f} lb: {3} \n{1}".format(i, trn_ds.extract_sentence(ask_id), score, lb))
        trn_ds.update(ask_id, lb)

        insertion_order.append(ask_id)
        if i % max(1, min(quota, len(unlabeled_idxs_orig))/10) == 0:
            query_num.append(i)
            E_out.append(final_score(trn_ds.extract_labeled_dataframe()))

    if return_ins_order:
        return query_num, E_out, insertion_order
    return query_num, E_out


def insert_in_batch_AL(trn_ds, final_score, lbr, ss_type, labeled_df=None, quota=5000, batch_num=20,
                       return_ins_order=False):
    # type: (TextDataset, callable, Labeler, SynState, int, bool) -> (list, list)
    E_out = []
    insertion_order = []
    trn_ds = copy.deepcopy(trn_ds)
    query_num = [0]
    E_out.append(final_score(trn_ds.extract_labeled_dataframe()))

    for i in range(0, quota, batch_num):
        if len(trn_ds.get_unlabeled_entries()) == 0:
            # print "first end"
            break  # finished labeling all examples
        df = trn_ds.extract_dataframe()
        cn.curr_train_df = trn_ds.extract_labeled_dataframe()
        unlabeled_idxs_orig = list(df[df[cn.tag_col].isnull()].index)
        if ss_type == "random":
            idx_score_list = map(lambda idx: (idx, random.random()), unlabeled_idxs_orig)
        else:
            idx_score_list = calculate_heuristic_bulk(df, cn.col_names, ss_type,
                                                      unlabeled_idxs_orig, labeled_df)
        for j in range(batch_num):
            if len(trn_ds.get_unlabeled_entries()) == 0:
                break  # finished labeling all examples
            cn.curr_train_df = trn_ds.extract_labeled_dataframe()
            ask_id, score = max(idx_score_list, key=lambda a: a[1])
            idx_score_list.remove((ask_id, score))

            lb = lbr.label(trn_ds.extract_sentence(ask_id))
            # print("{0}  scr: {2:.3f} lb: {3} \n{1}".format(i+j, trn_ds.extract_sentence(ask_id), score, lb))
            trn_ds.update(ask_id, lb)
            insertion_order.append(ask_id)
        # print("end batch\n")

        query_num.append(i+batch_num)  # update query-num ( actually this is i+j, but this way it is not consistent)
        E_out.append(final_score(trn_ds.extract_labeled_dataframe()))
        cn.last_E_out = E_out[-1]

    if return_ins_order:
        return query_num, E_out, insertion_order
    return query_num, E_out


def score_per_add_al(labeled_pool_df, base_training_df, validation_data_df):
    # type: (DataFrame, DataFrame, DataFrame) -> tuple

    gen_pool_df = labeled_pool_df.copy(deep=True)
    gen_pool_df[cn.col_names.tag] = [np.NaN] * len(gen_pool_df)  # clear all tags
    enriched_train_df = pd.concat([base_training_df, gen_pool_df], ignore_index=True)

    extractor = cn.Feature_Extractor(enriched_train_df, cn.col_names)  # build the feature extractor

    trn_ds = TextDataset(enriched_train_df, cn.col_names, extractor)

    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())

    ideal_df = pd.concat([base_training_df, labeled_pool_df], ignore_index=True)
    lbr = IdealTextLabeler(TextDataset(ideal_df, cn.col_names, extractor))

    scoring_fun = lambda ds: run_classifier(ds.extract_labeled_dataframe(), validation_data_df)
    ex_added_list, res_list = run_active_learning(trn_ds, scoring_fun, lbr, qs, len(enriched_train_df))  # label all df

    return ex_added_list, res_list


def scores_per_add_default(labeled_pool_df, base_training_df, validation_data_df):
    # type: (pd.DataFrame, pd.DataFrame, callable, int) -> (list, list)
    """
    get result from adding examples from $labeled_pool_df to $base_train_df, training them and seeing their results
    :param labeled_pool_df: df containing the new examples we asked the expert for
    :param base_train_df: the original training examples we had
    :param validation_data_df:  df containing the validation data we test our model's performance on
    :return: (ex_added_list, res_list), a list containing the number of examples added
                            and a list containing its corresponding result
    """
    # return score_per_add_al(labeled_pool_df, base_training_df, validation_data_df)
    return score_per_addition_results(labeled_pool_df, base_training_df,
                                      lambda en_df: run_classifier(en_df, validation_data_df))


def score_per_addition_results(labeled_pool_df, base_train_df, get_final_score):
    # type: (pd.DataFrame, pd.DataFrame, callable) -> (list, list)
    """
    get result from adding examples from $labeled_pool_df to $base_train_df, training them and seeing their results
    :param labeled_pool_df: df containing the new examples we asked the expert for
    :param base_train_df: the original training examples we had
    :param get_final_score: a function that returns the 'result' we want using our current enriched training data
    :return: (ex_added_list, res_list), a list containg the number of examples added and a list containing its corresponding result
    """
    ex_added_list = range(0, len(labeled_pool_df) + 1)
    enriched_train_data_df = pd.concat([base_train_df, labeled_pool_df])  # start with the base training data
    res_list = map(lambda ex_add: get_final_score(enriched_train_data_df[:len(base_train_df) + ex_add]), ex_added_list)
    return ex_added_list, res_list