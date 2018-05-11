from ResearchNLP import Constants as cn
from ResearchNLP.knowledge_bases import kb_helper, KnowledgeBase
from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools.text_mod_options import ModOption, \
    get_modification_options
from ResearchNLP.text_synthesis.synthesis_algorithms.basic_tools.text_modop import TextModOp
from ResearchNLP.util_files.NLP_utils import pos_tagger
from ResearchNLP.util_files.column_names import get_prev_states_from_sent
from ResearchNLP.util_files.combinatorics_util import flatten_lists


def build_modification_operators_from_mod_option(sent, mod_opt, sem_dist, k_base=None):
    # type: (str, ModOption, int, KnowledgeBase) -> list

    if len(mod_opt.word) == 1:
        return []  # don't switch single chars
    if cn.verbose: print 'switch word: ' + mod_opt.word
    k_base = k_base if k_base is not None else kb_helper.k_base
    pos_list_orig = pos_tagger.pos_tag_sent(sent, only_hash=True)

    s_words = k_base.select_words_in_dist(sent, mod_opt.word, sem_dist)

    def does_keep_pos(text_modop, pos_list_old):  # filter by ensuring the same pos in both old and new
        # type: (TextModOp, list) -> bool
        pos_list_new = pos_tagger.pos_tag_sent(text_modop.apply(), only_hash=True)
        return pos_list_new == pos_list_old  # new sentence is in the same semantic structure as the old one

    # create a TextModOp for each word in the semantic-distance
    return filter(lambda text_modop:  # not the same sent, not switching single char (like in It'[s])
                  text_modop.apply().lower() != sent.lower() and len(text_modop.word) != 1
                  and does_keep_pos(text_modop, pos_list_orig),
                  # filter by ensuring the same pos in both old and new
                  map(lambda d_word: TextModOp(mod_opt.orig_sent, mod_opt.word, mod_opt.word_idx, d_word), s_words))


mod_op_sum = 0
call_count = 0


def synthesize_mod_operators_sent(sent, text_df, col_names, mod_options=None):
    # type: (str, int, list, DataFrame) -> list
    if cn.verbose: print sent
    sem_dist = cn.sem_dist
    global call_count, mod_op_sum
    call_count += 1

    if mod_options is None:
        mod_options = get_modification_options([sent], text_df, col_names)[0][1]

    retval = flatten_lists(map(lambda opt: build_modification_operators_from_mod_option(sent, opt, sem_dist),
                               mod_options))
    mod_op_sum += len(retval)

    return retval


def synthesize_mod_operators_bulk(base_sents, text_df, col_names):
    # type: (list, int, bool) -> dict
    """
    generates possible modification operators based on $base_sents
    :param base_sents: list of base sents, from which we generate new ones
    :param text_df: a DataFrame containing all the sentences we currently work with
    :param col_names: ColumnNames object to complement text_df
    :return: dictionary from sent to all it's modification operators (TextModOp)
    """
    return dict(
        map(lambda (sent, mod_options): (sent, synthesize_mod_operators_sent(sent, text_df, col_names, mod_options)),
            # list of (sent, list of TextModOp), for each sentence his modif operators, in bulk if we want a human
            # better in map because of cache in in-code parallelism
            get_modification_options(base_sents, text_df, col_names))
    )


def synthesize_tree_depth1_bulk(base_sents, text_df, col_names):
    sents_mod_ops = synthesize_mod_operators_bulk(base_sents, text_df, col_names)  # takes most of time
    new_tuples = []
    for (orig_sent, mod_ops) in sents_mod_ops.iteritems():
        # new_tuples += map(lambda new_sent: (new_sent, [new_sent, orig_sent]), TextModOp.apply_mod_ops(mod_ops))
        orig_sent_history = get_prev_states_from_sent(text_df, col_names, orig_sent)
        new_tuples += map(lambda new_sent: (new_sent, [new_sent, orig_sent] + orig_sent_history[1:]),
                          TextModOp.apply_mod_ops(mod_ops))
    # print "syn_depth1 bunk ", len(new_tuples)
    return new_tuples
