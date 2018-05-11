from libact.base.interfaces import QueryStrategy
from libact.models import SVM
from libact.query_strategies import UncertaintySampling, ActiveLearningByLearning, QUIRE, HintSVM
from pandas import DataFrame

from ResearchNLP.text_synthesis.heuristic_functions.heuristics.al_heuristics import SynStateALHeuristic
from ResearchNLP.util_files import ColumnNames
from ResearchNLP.util_files.libact_utils import TextDataset

all_heuristics = [("ALBL", lambda: SynStateALBL)]


class SynStateALBL(SynStateALHeuristic):
    @staticmethod
    def build_query_strategy(sent_df, col_names):
        # type: (DataFrame, ColumnNames) -> QueryStrategy
        """
        Builds and returns a QueryStrategy
            using a feature extractor and a base_df
        """
        init_extractor = SynStateALHeuristic.build_feature_extractor(sent_df, col_names)
        combined_features = init_extractor.transform(sent_df, col_names)
        trn_ds = TextDataset(sent_df, col_names, None, features=combined_features)
        return ActiveLearningByLearning(trn_ds,
                                        query_strategies=[
                                            UncertaintySampling(trn_ds,
                                                                model=SVM(C=100, gamma=3.1, kernel='rbf',
                                                                          decision_function_shape='ovr')),
                                            QUIRE(trn_ds),
                                            HintSVM(trn_ds, cl=1.0, ch=1.0),
                                        ],
                                        T=1000,
                                        uniform_sampler=True,
                                        model=SVM(C=100, gamma=3.1, kernel='rbf', decision_function_shape='ovr')
                                        )
