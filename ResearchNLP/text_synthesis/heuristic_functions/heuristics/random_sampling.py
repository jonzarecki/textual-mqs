import random

from ResearchNLP.text_synthesis.heuristic_functions.synthesis_state import SynState

all_heuristics = [("random-score", lambda: SynStateRandom)]


class SynStateRandom(SynState):
    def __init__(self, state_idx, sent_df, col_names, prev_state=None):
        # type: (int, DataFrame, ColumnNames, SynState) -> None
        super(SynStateRandom, self).__init__(state_idx, sent_df, col_names, prev_state)

    def get_state_score(self):
        # type: () -> float
        """ Extracts the active-learning score and retrieve it """
        if self.prev_state is None:
            return 0  # initial state score
        return random.random()  # random score between 0 and 1
