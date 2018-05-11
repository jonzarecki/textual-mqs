import random
from functools import partial

import pandas as pd
from pandas import DataFrame

from ResearchNLP import Constants as cn
from ResearchNLP.text_synthesis.heuristic_functions import SynStateUncertainty, SynStateTrainDataGain
from ResearchNLP.text_synthesis.heuristic_functions.heuristic_fun_helper import calculate_heuristic_bulk
from ResearchNLP.util_files import ColumnNames, combinatorics_util


def choose_best_by_uncertainty(sent_df, col_names, count, sent_pool):
    # type: (DataFrame, ColumnNames, int) -> list
    return _choose_best_by_heuristic_fun(sent_df, col_names, count, SynStateUncertainty, sent_pool)


def choose_best_by_train_data_gain(sent_df, col_names, count, sent_pool):
    # type: (DataFrame, ColumnNames, int) -> list
    return _choose_best_by_heuristic_fun(sent_df, col_names, count, SynStateTrainDataGain, sent_pool)


def choose_best_randomly(sent_df, col_names, count, sent_pool):
    # type: (DataFrame, ColumnNames, int) -> list
    unlabeled_idxs = filter(lambda idx: sent_df[col_names.text][idx] not in sent_pool,
                            pd.np.where(sent_df[col_names.tag].isnull())[0])
    return [sent_df[col_names.text][random.choice(unlabeled_idxs)] for _ in range(count)]


# 2 options to call, or using the function names above, or by calling the template
def choose_best_by_heuristic_template(ss_type):
    return partial(lambda sent_df, col_names, count, sent_pool:
                   _choose_best_by_heuristic_fun(sent_df, col_names, count, ss_type, sent_pool))


def _choose_best_by_heuristic_fun(sent_df, col_names, count, ss_type, sent_pool):
    # type: (DataFrame, ColumnNames, int, SynState) -> list
    cn.add_experiment_param(ss_type.__name__)

    unlabeled_idxs = pd.np.where(sent_df[col_names.tag].isnull())[0]
    idx_text_col = list(sent_df[col_names.text][unlabeled_idxs].iteritems())
    filtered_tpls = filter(lambda (idx, s): s not in sent_pool, idx_text_col)
    filtered_idxs = map(lambda (idx, s): idx, filtered_tpls)
    assert len(filtered_idxs) >= count, "Not enough unlabeled instances to choose from (after filtering)"

    score_idx_list = calculate_heuristic_bulk(sent_df, col_names, ss_type, filtered_idxs)

    return _choose_by_heuristic_score_diverse_origins(sent_df, col_names, count, score_idx_list)


def _choose_by_heuristic_score_diverse_origins(sent_df, col_names, count, score_idx_list):
    sent_pool = set()
    assert len(score_idx_list) >= count, "Not enough instances to choose from"

    all_origins = set(map(lambda (u_idx, score): sent_df[col_names.prev_states][u_idx][-1], score_idx_list))
    origins_pool = set()
    chosen_idxs = set()
    origins_col = map(lambda hist: hist[-1], sent_df[col_names.prev_states].tolist())
    text_col = sent_df[col_names.text].tolist()
    filtered_idx_score_list = list(score_idx_list)
    while len(chosen_idxs) < count:  # generated count sentences
        filtered_idx_score_list = filter(lambda (u_idx, _): origins_col[u_idx] not in origins_pool and
                                                            text_col[u_idx] not in sent_pool, filtered_idx_score_list)
        if len(filtered_idx_score_list) == 0 or len(origins_pool) == len(all_origins):  # all origins used
            origins_pool = set()
            filtered_idx_score_list = filter(lambda (u_idx, _): origins_col[u_idx] not in origins_pool and
                                                                text_col[u_idx] not in sent_pool, score_idx_list)

        u_idx, score = filtered_idx_score_list[combinatorics_util.weighted_random_choice(map(lambda a: a[1] ** 3,
                                                                                             filtered_idx_score_list))]
        inst_origin = sent_df[col_names.prev_states][u_idx][-1]
        if inst_origin not in origins_pool and text_col[u_idx] not in sent_pool:
            origins_pool.add(inst_origin)
            sent_pool.add(text_col[u_idx])
            chosen_idxs.add(u_idx)
            score_idx_list.remove((u_idx, score))
    return list(chosen_idxs)
