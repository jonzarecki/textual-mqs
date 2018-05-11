import pandas as pd
from pandas import DataFrame

from ResearchNLP.text_synthesis.heuristic_functions import *
from ResearchNLP.text_synthesis.heuristic_functions.choose_best_for_addition import \
    _choose_by_heuristic_score_diverse_origins
from ResearchNLP.util_files import combinatorics_util, ColumnNames


def select_from_pool_uncertainty(gen_pool_df, base_training_df, col_names, sent_pool, count=1):
    return _select_from_pool_using_heuristic(gen_pool_df, base_training_df, col_names,
                                             count, SynStateUncertainty, sent_pool)


def select_from_pool_randomly(gen_pool_df, base_training_df, col_names, count=1, sent_pool=None):
    return combinatorics_util.weighted_random_choice_bulk([1.0] * len(gen_pool_df), count)


def _select_from_pool_using_heuristic(gen_pool_df, base_training_df, col_names, count, ss_type, sent_pool):
    # type: (DataFrame, DataFrame, ColumnNames, SynState) -> DataFrame
    gen_pool_df = gen_pool_df[gen_pool_df[col_names.text].isin(sent_pool) == False]  # not in sent_pool
    enriched_train_df = pd.concat([gen_pool_df, base_training_df], ignore_index=True)
    heuristic_fun = prepare_heuristic_fun(enriched_train_df, col_names)
    idx_scores_list = map(lambda idx: (idx, heuristic_fun(idx, ss_type)), range(len(gen_pool_df)))

    return _choose_by_heuristic_score_diverse_origins(enriched_train_df, col_names, count, idx_scores_list)
