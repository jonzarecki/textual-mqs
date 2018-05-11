from abc import ABCMeta, abstractmethod

import pandas as pd
from pandas import DataFrame
from six import with_metaclass

from ResearchNLP import Constants as cn
from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools import text_modop
from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools.text_modops_util import synthesize_mod_operators_bulk
from ResearchNLP.util_files import ColumnNames
from ResearchNLP.util_files.column_names import extend_sent_history, add_sentences_and_histories_to_df, \
    prepare_df_columns


class SynState(with_metaclass(ABCMeta, object)):
    def __init__(self, state_idx, sent_df, col_names, prev_state=None):
        # type: (int, DataFrame, ColumnNames, SynState) -> None
        self.text_state = sent_df[col_names.text][state_idx]
        self.state_idx = state_idx
        self.sent_df = sent_df
        self.col_names = col_names
        self.prev_state = prev_state
        if self.prev_state is None:
            self.orig_state = self
        else:
            self.orig_state = self.prev_state.orig_state
            assert type(prev_state).__name__ == type(self).__name__, \
                "previous state should be the same type as current state"

    @abstractmethod
    def get_state_score(self):
        # type: () -> float
        """ Calculates the heuristic function's score and retrieves it
        The bigger the score the better, normalized between (0,1) """
        pass

    def build_next_sent_df(self):
        # type: (int) -> pd.DataFrame
        mod_ops = synthesize_mod_operators_bulk([self.text_state], self.sent_df, self.col_names)[self.text_state]
        new_sents = text_modop.TextModOp.apply_mod_ops(mod_ops)
        if len(new_sents) != 0:
            curr_history = extend_sent_history(self.sent_df, self.col_names, [self.text_state])
            new_tuples = map(lambda s: (s, [s] + curr_history), new_sents)
            combined_df = add_sentences_and_histories_to_df(self.sent_df, self.col_names, new_tuples)
            combined_df = prepare_df_columns(combined_df, self.col_names)
        else:
            combined_df = self.sent_df
        cn.total_mod_ops += 1
        return combined_df

    def get_next_states(self):
        combined_df = self.build_next_sent_df()
        base_len = len(self.sent_df)
        return map(lambda (idx, text_state): type(self)(base_len + idx, combined_df,
                                                        self.col_names, prev_state=self),
                   enumerate(combined_df[base_len:][self.col_names.text]))

    def get_prev_states_list(self):
        curr_state = self
        prev_states = [self.text_state]
        while curr_state.prev_state is not None:
            curr_state = curr_state.prev_state
            prev_states.append(curr_state.text_state)
        return prev_states
