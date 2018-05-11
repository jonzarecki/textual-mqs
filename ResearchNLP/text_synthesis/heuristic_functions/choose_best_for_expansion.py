import random
import time

import numpy as np
import pandas as pd

from ResearchNLP.text_synthesis.heuristic_functions import SynStateUncertainty

# order = map(lambda a: random.randint(1, 1000), range(1000))
sa = time.time()
counter = 0


def choose_random_sents_from_df(sent_df, col_names, count):
    r = np.random.RandomState((counter, len(sent_df), count))
    global counter, sa
    if time.time() - sa > 10.0:
        sa = time.time()
        counter = 0
    counter += 1
    # return [sent_df[col_names.text][order[(i + counter) % 1000] % len(sent_df)] for i in range(count)]
    return [sent_df[col_names.text][r.choice(range(len(sent_df)))] for _ in range(count)]
    # return [sent_df[col_names.text][random.choice(range(len(sent_df)))] for _ in range(count)]


def choose_best_using_random_diverse_origins(sent_df, col_names, count):
    # type: (DataFrame, ColumnNames, int) -> list

    score_idx_dict = dict(map(lambda idx: (idx, random.random()), sent_df.index))

    from libact.base.interfaces import QueryStrategy
    score_idx_dict = QueryStrategy._port_dict_to_0_1_range(score_idx_dict)
    score_idx_list = list(score_idx_dict.iteritems())

    from ResearchNLP.text_synthesis.heuristic_functions.choose_best_for_addition import \
        _choose_by_heuristic_score_diverse_origins
    chosen_idxs = _choose_by_heuristic_score_diverse_origins(sent_df, col_names, count, score_idx_list)
    return map(lambda i: sent_df[col_names.text][i], chosen_idxs)


def choose_best_using_leave_out_uncertainty(sent_df, col_names, count):
    # type: (DataFrame, ColumnNames, int) -> list

    labeled_idxs = pd.np.where(sent_df[col_names.tag].notnull())[0]
    unlabeled_idxs = pd.np.where(sent_df[col_names.tag].isnull())[0]

    def leave_one_out_uncertainty(idx):
        sent_df2 = sent_df.copy(deep=True)
        sent_df2.loc[idx, col_names.tag] = None
        sent_qs2 = SynStateUncertainty.build_query_strategy(sent_df2, col_names)
        sent_qs2.get_score(idx)  # leave-one-out score
        return sent_qs2.real_scores_dict[idx]  # non-normalized score

    score_idx_dict = dict(map(lambda idx: (idx, leave_one_out_uncertainty(idx)), labeled_idxs))

    from libact.base.interfaces import QueryStrategy
    score_idx_dict = QueryStrategy._port_dict_to_0_1_range(score_idx_dict)
    score_idx_list = list(score_idx_dict.iteritems())
    if len(unlabeled_idxs) != 0:
        sent_qs = SynStateUncertainty.build_query_strategy(sent_df, col_names)
        score_idx_list += map(lambda idx: (idx, sent_qs.get_score(idx)), unlabeled_idxs)

    from ResearchNLP.text_synthesis.heuristic_functions.choose_best_for_addition import \
        _choose_by_heuristic_score_diverse_origins
    chosen_idxs = _choose_by_heuristic_score_diverse_origins(sent_df, col_names, count, score_idx_list)
    return map(lambda i: sent_df[col_names.text][i], chosen_idxs)
