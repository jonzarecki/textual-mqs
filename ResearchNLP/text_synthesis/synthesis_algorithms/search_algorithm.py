from functools import partial

import numpy as np
from simpleai.search import beam, hill_climbing_stochastic, hill_climbing
from simpleai.search.local import hill_climbing_weighted_stochastic

from ResearchNLP.text_synthesis.synthesis_algorithms.search_problem import BestInstanceProblem
from ResearchNLP.text_synthesis.synthesis_algorithms.synthesis_algorithm import SynthesisAlgorithm
from ResearchNLP.util_files.column_names import extend_sent_history

search_alg_codes = ["DHC", "SHC", "SWHC", "BS"]


class LocalSearchAlgorithm(SynthesisAlgorithm):
    def __init__(self, alg_code, iterations_limit=3, beam_size=4):
        # type: (str, int) -> None
        """

        :param alg_code: the type-code of the search algorithm we'll be using, choose between:
                        - DHC : deterministic hill climbing
                        - SHC : stochastic hill climbing
                        - SWHC : stochastic weighted hill climbing
                        - BS : beam search
        :param iterations_limit: the limit of iterations for the local search algorithm
        :param beam_size: the size of the beam if the algorithm is beam search
        """
        assert alg_code in search_alg_codes, "LocalSearchAlgorithm: alg code incorrect, not in search_alg_codes list"
        # actual iteration limit
        iterations_limit = min(np.random.geometric(0.25), iterations_limit)

        self.alg_code = alg_code
        self.iterations_limit = iterations_limit
        if alg_code == "DHC":
            self.search_alg = partial(hill_climbing, iterations_limit=iterations_limit)
        elif alg_code == "SHC":
            self.search_alg = partial(hill_climbing_stochastic, iterations_limit=iterations_limit)
        elif alg_code == "SWHC":
            self.search_alg = partial(hill_climbing_weighted_stochastic, iterations_limit=iterations_limit)
        elif alg_code == "BS":
            self.search_alg = partial(beam, iterations_limit=iterations_limit, beam_size=beam_size)
            self.beam_size = beam_size
        else:
            assert False, "LocalSearchAlgorithm: " + alg_code + "function not implemented"

    def run_alg(self, text_df, col_names, init_state):
        problem = BestInstanceProblem(text_df, col_names, init_state)
        new_state = self.search_alg(problem).state  # get the new state after running the algorithm
        return new_state.text_state, extend_sent_history(text_df, col_names, new_state.get_prev_states_list())

