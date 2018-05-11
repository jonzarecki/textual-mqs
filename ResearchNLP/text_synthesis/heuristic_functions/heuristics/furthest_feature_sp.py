import numpy as np
from numpy import linalg as LA
from pandas import DataFrame

from ResearchNLP import Constants as cn
from ResearchNLP.text_synthesis.heuristic_functions.synthesis_state import SynState
from ResearchNLP.util_files import ColumnNames

all_heuristics = [("furthest-feature-sp", lambda: SynStateFurthestFeatureSp)]


class SynStateFurthestFeatureSp(SynState):
    """
    Search space with a Furthest-away heuristic, at each state we try and make the most dissimilar new sentence,
        so the score is higher for least similar sentences
    """

    def __init__(self, state_idx, sent_df, col_names, prev_state=None):
        # type: (int, DataFrame, ColumnNames, SynStateFurthestFromPos) -> None
        super(SynStateFurthestFeatureSp, self).__init__(state_idx, sent_df, col_names, prev_state)

        relevant_idxs = list(self.sent_df[self.sent_df[col_names.tag].notnull()].index)
        # relevant_idxs.remove(state_idx)
        # relevant_idxs = np.array(relevant_idxs)
        if self.col_names.feature_repr in self.sent_df.columns:
            self.sent_repr = self.sent_df[self.col_names.feature_repr][self.state_idx]
            self.min_dist = np.min(map(lambda (i, r): LA.norm(r[col_names.feature_repr] - self.sent_repr, 1) ** 2,
                                       self.sent_df.iloc[relevant_idxs].iterrows()), axis=0)
        else:
            extractor = cn.Feature_Extractor(sent_df, col_names)  # this is actually faster than DataFrame.append()
            X_all = extractor.transform(sent_df, col_names)
            self.sent_repr = X_all[self.state_idx]

            self.min_dist = np.min(map(lambda feat: LA.norm(feat - self.sent_repr, 1) ** 2,
                                       X_all[relevant_idxs]))

    def get_state_score(self):
        # type: () -> float
        """ Extracts the similarity score and retrieve it
            Assume that we want to be as far away as possible from orig (minimum similarity) """
        if self.prev_state is None:
            return 0  # initial state score
        return self.min_dist
