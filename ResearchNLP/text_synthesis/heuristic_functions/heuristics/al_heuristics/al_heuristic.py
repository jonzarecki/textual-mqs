from libact.base.interfaces import QueryStrategy
from pandas import DataFrame

from ResearchNLP import Constants as cn
from ResearchNLP.text_synthesis.heuristic_functions.synthesis_state import SynState
from ResearchNLP.util_files import ColumnNames


class SynStateALHeuristic(SynState):
    def __init__(self, state_idx, sent_df, col_names, prev_state=None):
        # type: (int, DataFrame, ColumnNames, SynState) -> None

        super(SynStateALHeuristic, self).__init__(state_idx, sent_df, col_names, prev_state)
        self.sent_qs = None
        if self.prev_state is not None:
            self.sent_qs = self.prev_state.build_next_states_qs(sent_df)
        self.next_states_qs = None

    @staticmethod
    def build_query_strategy(sent_df, col_names):
        # type: (DataFrame, ColumnNames) -> QueryStrategy
        """
        Builds and returns a QueryStrategy
            using a feature extractor and a base_df
        """
        raise NotImplementedError

    @staticmethod
    def build_feature_extractor(sent_df, col_names):
        # type: (pd.DataFrame, ColumnNames) -> tuple(FeatureExtractor, pd.DataFrame)
        """
        Builds and returns a feature extractor using sent_df
        """
        return cn.Feature_Extractor(sent_df, col_names)  # build the feature extractor

    def get_state_score(self):
        # type: () -> float
        """ Extracts the active-learning score and retrieve it """
        if self.prev_state is None:
            return 0  # initial state score
        return self.sent_qs.get_score(self.state_idx)

    def build_next_states_qs(self, combined_df):
        """ Build once, only when needed """
        if self.next_states_qs is None:
            # use one query-strat for all next states
            self.next_states_qs = self.build_query_strategy(combined_df, self.col_names)
        return self.next_states_qs
