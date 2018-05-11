import pandas as pd
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from ResearchNLP import Constants as cn
from ResearchNLP.classification_experts import human_expert_gui
from ResearchNLP.util_files.NLP_utils import stem_tokenize, pos_tagger


class ModOption(object):
    def __init__(self, orig_sent, word, word_idx):
        self.orig_sent = orig_sent
        self.word = word
        self.word_idx = word_idx


def is_token_word_in_wordpool(word):
    # type: (basestring) -> bool
    """
    filter function, returns True if the tokenized form of $word is in Constant's wordpool
    and false otherwise
    :param word: the word we want to check
    :return: boolean as stated above
    """
    try:
        word_token = stem_tokenize(word)  # weird single words may throw in this situation
    except IndexError:
        return False
    return (stem_tokenize(word)[0] in cn.tok_word_pool) if word_token else False


def get_switched_words_for_sent(sent, sent_pool_df, col_names):
    sent_idx = sent_pool_df[col_names.text].values.tolist().index(sent)
    orig_sent = sent_pool_df[col_names.prev_states][sent_idx][-1]
    switch_count = 0
    switched_words = []
    tokenized_orig_sent = pos_tagger.tokenize_sent(orig_sent)
    tokenized_sent = pos_tagger.tokenize_sent(sent)
    assert len(tokenized_sent) == len(tokenized_orig_sent), "switched sentences should have the same number of words"
    for i, (orig_word, curr_word) in enumerate(zip(tokenized_orig_sent, tokenized_sent)):
        if orig_word != curr_word:  # word from curr sent did not appear in orig sent
            switched_words.append((i, curr_word))
            switch_count += 1
    assert len(switched_words) == len(sent_pool_df[col_names.prev_states][sent_idx]) - 1, \
        "each switch should result in one switched words"
    return switched_words


def get_switch_word_options(sent, text_df, col_names):
    # type: (str, list) -> list
    """
    returns all switch word options for the given sentence
    :param sent: a sentence
    :param text_df: a DataFrame containing all the sentences we currently work with
    :param col_names: ColumnNames object to complement text_df
    :return: list of ModOption
    """
    # cn.add_experiment_param('with_switches')
    tagged_sent = pos_tagger.pos_tag_sent(sent)
    # sent_switches = get_switched_words_for_sent(sent, text_df, col_names) #  (i, word) not in sent_switches and
    mod_options = []
    for i, (word, tag, st_idx) in enumerate(tagged_sent):
        if (tag == "VERB" or tag == "ADJ" or tag == "NOUN") \
                and word.isalnum() and word.lower() not in ENGLISH_STOP_WORDS:
            # tok_word = pos_tagger.tokenize_sent(word)
            # if len(tok_word) == 1 and (i, tok_word[0]) not in sent_switches:
            mod_options.append(ModOption(sent, word, st_idx))
    return mod_options


def get_modification_options(base_sents, text_df, col_names):
    # type: (list, bool) -> list
    """
    generates new examples based on $base_sents
    :param base_sents: list of base sents, from which we generate new ones
    :return: list of (sent, ModOption), for each sentence all it's modification options
    """
    sent_mod_options = [None] * len(base_sents)  # list of (sent, list of ModOption), for each sentence his ModOptions

    # choose switch word with human expert
    if cn.choose_sword_with_expert:
        # put all generated sentences in a new DataFrame to choose their switch word using a human expert
        switch_word_df = pd.DataFrame(columns=[cn.text_col, cn.switch_word_column])
        switch_word_df[cn.text_col] = base_sents  # fill the text column with the new sentences

        sent_with_base_word_df = \
            human_expert_gui.choose_most_important_word_by_expert(switch_word_df, cn.text_col, cn.switch_word_column)
        sent_with_base_word_df.to_pickle('sents_with_switch_word.pkl')  # save choices to file

        for (row_idx, row) in sent_with_base_word_df.iterrows():
            sent = row[1][cn.text_col]
            word = row[1][cn.switch_word_column]
            word_idx = sent.find(word)  # assume we are talking on first appearance, no other choice
            assert word.find(" ") == -1, "switch word shouldn't contain spaces"  # in the future this can change
            assert word_idx != -1, "the switch word must be in the sentence"
            sent_mod_options[row_idx] = (sent, [ModOption(sent, word, word_idx)])  # list of 1

    # choose switch word with rules
    else:
        for i, sent in enumerate(base_sents):
            sent_mod_options[i] = (sent, get_switch_word_options(sent, text_df, col_names))

    return sent_mod_options
