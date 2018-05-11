from libact.base.interfaces import QueryStrategy
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling
from pandas import DataFrame

from ResearchNLP.text_synthesis.heuristic_functions.heuristics.al_heuristics import SynStateALHeuristic
from ResearchNLP.util_files import ColumnNames
from ResearchNLP.util_files.libact_utils import TextDataset

all_heuristics = [("uncertainty lc LogReg", lambda: SynStateUncertainty)]


class SynStateUncertainty(SynStateALHeuristic):
    @staticmethod
    def build_query_strategy(sent_df, col_names):
        # type: (DataFrame, ColumnNames) -> QueryStrategy
        """
        Builds and returns a QueryStrategy
            using a feature extractor and a base_df
        """
        init_extractor = SynStateALHeuristic.build_feature_extractor(sent_df, col_names)
        combined_features = init_extractor.transform(sent_df, col_names)
        return UncertaintySampling(TextDataset(sent_df, col_names, None, features=combined_features),
                                   method='lc', model=LogisticRegression())
