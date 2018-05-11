import pandas as pd
from pandas import DataFrame

from ResearchNLP.text_synthesis.heuristic_functions.synthesis_state import SynState
from ResearchNLP.util_files import ColumnNames, pandas_util

all_heuristics = [("test-data-gain", lambda: SynStateTestDataGain)]


class SynStateTestDataGain(SynState):
    """
    Returns a score based on the increase of the test data accuracy,
    cheating but good for testing heuristics as it is close to the perfect heuristic
    """

    def __init__(self, state_idx, sent_df, col_names, prev_state=None):
        # type: (int, DataFrame, ColumnNames, SynStateTestDataGain) -> None

        super(SynStateTestDataGain, self).__init__(state_idx, sent_df, col_names, prev_state)
        self.labeled_state_df = None
        if prev_state is not None:
            labeled_df = prev_state.build_next_states_labeled_df(sent_df)
            assert len(labeled_df) == len(sent_df)
            self.labeled_state_df = pandas_util.extract_row_from_df(labeled_df, state_idx)
            self.curr_train_df = self.sent_df[self.sent_df[self.col_names.tag].notnull()].reset_index(drop=True)
        self.next_states_labeled_df = None

    def get_state_score(self):
        # type: () -> float
        """ adds the state's textual state to the dataset and check the increase in accuracy"""
        if self.prev_state is None:
            return 0  # initial state score
        from ResearchNLP.z_experiments.experiment_util import run_classifier
        from ResearchNLP import Constants as cn

        # if hasattr(cn, "last_E_out"):
        #     before = cn.last_E_out
        # else:
        before = run_classifier(self.curr_train_df, cn.validation_data_df).acc

        combined_df = pd.concat([self.curr_train_df, self.labeled_state_df], ignore_index=True)
        after = run_classifier(combined_df, cn.validation_data_df).acc

        diff = after - before  # difference in acc score. NOT NORMALIZED, but its supposed to be OK
        return diff

    @staticmethod
    def label_dataframe_with_expert(sent_df, col_names, unlabeled_df_labeled=None):
        # type: (DataFrame, ColumnNames, DataFrame) -> DataFrame
        """ Label sent_df with the expert """
        unlabeled_df = sent_df[sent_df[col_names.tag].isnull()].copy(deep=True)
        if unlabeled_df_labeled is None:
            from ResearchNLP.z_experiments.experiment_util import label_df_with_expert
            unlabeled_df_labeled = label_df_with_expert(unlabeled_df, col_names, print_status=False)
        # assuming unlabeled_df_labeled is meant for the end of $sent_df
        unlabeled_df_labeled_st = len(sent_df) - len(unlabeled_df_labeled)
        df_len = len(sent_df)
        for un_idx in unlabeled_df.index:
            assert unlabeled_df_labeled_st <= un_idx < df_len, \
                "$unlabeled_df_labeled should hold info for the end of the dataframe"
        labeled_sent_df = sent_df.copy(deep=True)
        for idx in unlabeled_df.index:  # set all new labels
            labeled_sent_df.iloc[idx] = unlabeled_df_labeled.iloc[idx - unlabeled_df_labeled_st].copy(deep=True)
        return labeled_sent_df

    def build_next_states_labeled_df(self, combined_df):
        """ Build once, only when needed """
        if self.next_states_labeled_df is None:
            # label once for all new sents
            self.next_states_labeled_df = self.label_dataframe_with_expert(combined_df, self.col_names)
        return self.next_states_labeled_df
