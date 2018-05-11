import random

from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools.text_modops_util import synthesize_mod_operators_sent
from ResearchNLP.text_synthesis.synthesis_algorithms.synthesis_algorithm import SynthesisAlgorithm
from ResearchNLP.util_files.column_names import extend_sent_history, add_sentences_and_histories_to_df


class RandomAlgorithm(SynthesisAlgorithm):
    def run_alg(self, text_df, col_names, init_state):
        def apply_random_modification_operator(text_df, col_names, curr_state, curr_history):
            mod_ops = synthesize_mod_operators_sent(curr_state, text_df, col_names)
            if len(mod_ops) == 0:
                return curr_state, curr_history  # returning orig sent doesn't count

            new_sent = random.choice(mod_ops).apply()
            return new_sent, extend_sent_history(text_df, col_names, [new_sent, curr_state] + curr_history[1:])

        text_df = text_df.copy(deep=True)
        iter_lim = 1  # random.randint(1, 10)
        # iter_lim = min(np.random.geometric(0.2), 10)
        # iter_lim = 100
        curr_state = init_state
        curr_history = []
        for i in range(iter_lim):
            new_state, new_history = apply_random_modification_operator(text_df, col_names, curr_state, curr_history)
            if new_state == curr_state:
                break
            curr_state = new_state
            curr_history = new_history
            text_df = add_sentences_and_histories_to_df(text_df, col_names, [(curr_state, curr_history)])
        return curr_state, curr_history
