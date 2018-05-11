from abc import ABCMeta, abstractmethod

from pandas import DataFrame
from six import with_metaclass

from ResearchNLP.util_files import ColumnNames
from ResearchNLP.util_files.multiproc_util import parmap


class SynthesisAlgorithm(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def run_alg(self, text_df, col_names, init_state):
        # type: (DataFrame, ColumnNames, basestring) -> basestring
        """
        :param text_df: a DataFrame containing all the sentences we currently work with
        :param col_names: ColumnNames object to complement text_df
        :param init_state: the initial state from which we synthesize
        :return: a new textual state generated using the algorithm,
                 and all the previous textual state on the way as a list
        """
        pass

    def run_alg_parallel(self, text_df, col_names, init_states):
        # type: (DataFrame, ColumnNames, list) -> list
        """
        runs "run_alg()" in parallel generating from each sentence in $init_states
        :param text_df: a DataFrame containing all the sentences we currently work with
        :param col_names: ColumnNames object to complement text_df
        :param init_states: the initial states from which we synthesize
        :return: list of new textual states and previous states generated from each sentence
        """
        assert isinstance(init_states, list)
        # return map(lambda sent: self.run_alg(text_df, col_names, sent), init_states)
        if len(init_states) == 1:
            return map(lambda sent: self.run_alg(text_df, col_names, sent), init_states)
        return parmap(lambda sent: self.run_alg(text_df, col_names, sent), init_states, nprocs=3)
