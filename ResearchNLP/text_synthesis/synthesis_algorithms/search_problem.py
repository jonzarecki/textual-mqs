import random

import pandas as pd
from simpleai.search import SearchProblem

from ResearchNLP import Constants as cn
# from ResearchNLP.text_synthesis.heuristic_functions.synthesis_state import SynState
from ResearchNLP.util_files import ColumnNames, pandas_util


class BestInstanceProblem(SearchProblem):
    def __init__(self, sent_df, col_names, init_text_state=None):
        # type: (pd.DataFrame, ColumnNames, str) -> None
        super(BestInstanceProblem, self).__init__()

        cn.inst_count += 1
        self.sent_pool_df = sent_df
        self.col_names = col_names

        cn.add_experiment_param(cn.ss_type.__name__)

        if init_text_state is not None:
            init_row = sent_df[sent_df[col_names.text] == init_text_state]
            assert len(init_row) != 0, "init_text_state not in send_df"
            # initial_state is used in BestInstanceProblem
            self.initial_state = cn.ss_type(init_row.index[0], self.sent_pool_df, col_names)
        self.init_states = None

    def actions(self, state):
        # type: (SynState) -> list
        """
        :param state: a Search state
        :return: list of SynState for each available action
        """
        return state.get_next_states()

    def result(self, state, action):
        # type: (SynState, SynState) -> SynState
        """
        As the action is a new SynState we'll just return it
        """
        return action

    def value(self, state):
        # type: (SynState) -> float
        score = state.get_state_score()
        # assert 0.0 <= score <= 1.0, "the score should be between 0.0 and 1.0"
        return score

    def generate_random_state(self):
        # type: () -> SynState
        if self.init_states is None:
            positive_df = pandas_util.all_positive_rows_df(self.sent_pool_df, self.col_names.tag, cn.pos_tags)
            self.init_states = map(lambda (idx, text_state): cn.ss_type(idx, self.sent_pool_df,
                                                                           self.col_names),
                                   positive_df[self.col_names.text].iteritems())
        return random.sample(self.init_states, 1)[0]  # not used in hill climbing
