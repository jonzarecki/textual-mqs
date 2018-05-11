import random

import numpy as np

from ResearchNLP import Constants as cn
from ResearchNLP.knowledge_bases import WordNetKB
from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools import text_mod_options
from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools.text_modop import TextModOp
from ResearchNLP.text_synthesis.synthesis_algorithms.synthesis_algorithm import SynthesisAlgorithm
from ResearchNLP.util_files.column_names import extend_sent_history

lecun_wordnet = None


def init_lecun_WordNet():
    global lecun_wordnet
    if lecun_wordnet is None:
        lecun_wordnet = WordNetKB(cn.distance_measure)


class LecunAugmentation(SynthesisAlgorithm):
    def __init__(self):
        init_lecun_WordNet()

    def run_alg(self, text_df, col_names, init_state):
        if len(cn.data_df[cn.data_df[col_names.text] == init_state]) == 0:  # only work with original sentences
            return init_state, None

        mod_options = text_mod_options.get_modification_options([init_state], text_df, col_names)[0][1]
        if len(mod_options) == 0:
            return init_state, None  # won't count as its the same sent

        r = min(np.random.geometric(0.5), len(mod_options))
        chosen_mod_options = random.sample(mod_options, r)
        curr_state = init_state
        for mod_opt in chosen_mod_options:  # apply all mod_options
            mod_opt.orig_sent = curr_state
            synset = lecun_wordnet._get_word_synset(curr_state, mod_opt.word)
            s_words = []
            if synset is not None:
                s_words = synset.lemma_names()
            mod_ops = map(lambda d_word: TextModOp(mod_opt.orig_sent, mod_opt.word, mod_opt.word_idx, d_word), s_words)
            # mod_ops = text_modops_util.build_modification_operators_from_mod_option(
            #     init_state, mod_opt, 0, lecun_wordnet)  # 0 for synonyms
            if len(mod_ops) != 0:
                s = min(np.random.geometric(0.5), len(mod_ops))
                curr_state = mod_ops[s-1].apply()
        return curr_state, extend_sent_history(text_df, col_names, [curr_state, init_state])
