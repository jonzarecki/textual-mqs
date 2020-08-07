import pandas as pd
from libact.query_strategies import RandomSampling

from ResearchNLP.text_synthesis.heuristic_functions import \
    SynStateRandom, SynStateTestDataGain, SynStateTrainDataGain, furthest_feature_sp, train_data_gain, \
    random_sampling, SynStateFurthestFeatureSp
from ResearchNLP.text_synthesis.heuristic_functions.heuristics.al_heuristics import \
    SynStateALHeuristic, hint_svm, quire, uncertainty_sampling
from ResearchNLP.util_files import ColumnNames
from ResearchNLP.util_files.combinatorics_util import flatten_lists
from ResearchNLP.util_files.libact_utils import TextDataset
from ResearchNLP.util_files.multiproc_util import parmap

latest_cnn = None

def prepare_heuristic_fun(enriched_train_df, col_names, labeled_df=None):
    # type: (pd.DataFrame, ColumnNames) -> callable
    # prepare beforehand for sharing
    shared_variables = dict()

    def heuristic_score_fun(inst_idx, ss_type):
        if ss_type == "Random":
            if "qs2" not in shared_variables:
                extractor = SynStateALHeuristic.build_feature_extractor(enriched_train_df, col_names)
                qs2 = RandomSampling(TextDataset(enriched_train_df, col_names, extractor))
                shared_variables["qs2"] = qs2
            qs2 = shared_variables["qs2"]
            return qs2.get_score(inst_idx)

        class Object(object):
            pass
        PS_type = type(ss_type.__name__, (object,), dict(orig_state=Object()))  # python hack for naming a type

        def prepare_prev_state(ss_type, prev_state=None):
            if prev_state is None:
                prev_state = PS_type()

            if issubclass(ss_type, SynStateALHeuristic):
                if str(ss_type)+"qs" not in shared_variables:
                    qs = ss_type.build_query_strategy(enriched_train_df, col_names)
                    shared_variables[str(ss_type)+"qs"] = qs
                qs = shared_variables[str(ss_type)+"qs"]
                prev_state.build_next_states_qs = lambda _: qs
             elif ss_type == SynStateTestDataGain:
                if "en_labeled_train_df" not in shared_variables:
                    enriched_labeled_train_df = SynStateTestDataGain. \
                        label_dataframe_with_expert(enriched_train_df, col_names, labeled_df)
                    shared_variables["en_labeled_train_df"] = enriched_labeled_train_df
                enriched_labeled_train_df = shared_variables["en_labeled_train_df"]
                prev_state.build_next_states_labeled_df = lambda _: enriched_labeled_train_df
            elif ss_type == SynStateRandom:
                pass  # return prev_state as it is
            return prev_state

        ss_prev_state = prepare_prev_state(ss_type)
        ss = ss_type(inst_idx, enriched_train_df, col_names, ss_prev_state)
        return ss.get_state_score()

    return heuristic_score_fun


def calculate_heuristic_bulk(enriched_df, col_names, ss_type, inst_idxs, labeled_df=None):
    heuristic_fun = prepare_heuristic_fun(enriched_df, col_names, labeled_df)
    if ss_type == SynStateTrainDataGain:
        # idx_score_list = map(lambda idx: (idx, heuristic_fun(idx, ss_type)), inst_idxs)
        idx_score_list = parmap(lambda idx: (idx, heuristic_fun(idx, ss_type)),
                            inst_idxs, chunk_size=10)
    elif ss_type == SynStateFurthestFeatureSp:
        # idx_score_list = map(lambda idx: (idx, heuristic_fun(idx, ss_type)), inst_idxs)
        idx_score_list = parmap(lambda idx: (idx, heuristic_fun(idx, ss_type)),
                                inst_idxs, chunk_size=25)
    else:
        idx_score_list = map(lambda idx: (idx, heuristic_fun(idx, ss_type)), inst_idxs)
    return idx_score_list


all_heuristics_list = flatten_lists([
   uncertainty_sampling.all_heuristics,
    quire.all_heuristics,
    hint_svm.all_heuristics,
    random_sampling.all_heuristics,
    train_data_gain.all_heuristics,
    furthest_feature_sp.all_heuristics
])


combined_heuristics_list = flatten_lists([
     uncertainty_sampling.all_heuristics,
    # quire.all_heuristics,
    hint_svm.all_heuristics,
    # random_sampling.all_heuristics,
    # train_data_gain.all_heuristics,
    # furthest_feature_sp.all_heuristics
])


def find_heuristic(heuristic_name):
    all_heuristics_names = map(lambda (n, p): n, all_heuristics_list)
    heuristic_idx = all_heuristics_names.index(heuristic_name)
    return all_heuristics_list[heuristic_idx][1]
