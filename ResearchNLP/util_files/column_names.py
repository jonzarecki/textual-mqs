import pandas as pd

from ResearchNLP.util_files import pandas_util
from ResearchNLP.util_files.NLP_utils import stem_tokenize


class ColumnNames(object):
    # helper class to keep the column names, saved here to avoid import loops in Constants
    def __init__(self, text, tag, tok_text, feature_repr, prev_states, sent_sim_repr):
        # type: (str, str, str, str, str, str) -> None
        self.text = text
        self.tag = tag
        self.tok_text = tok_text
        self.feature_repr = feature_repr
        self.prev_states = prev_states
        self.sent_sim_repr = sent_sim_repr  # obsolete

    def __iter__(self):
        return [self.text, self.tag, self.tok_text, self.feature_repr, self.prev_states, self.sent_sim_repr].__iter__()


def fix_sentences_encoding(sents):
    fixed_sents = []
    for s_orig in sents:
        s = s_orig
        try:
            s = s.encode("latin1")
        except:
            try:
                s = s.encode("utf-8")
            except:
                pass
        if type(s) != unicode:
            fixed_sents.append(unicode(s, errors="ignore"))
        else:
            fixed_sents.append(s)
    return fixed_sents


def prepare_df_columns(df, col_names):
    # type: (pd.DataFrame, ColumnNames) -> pd.DataFrame
    """
    initialized non-initialized columns in $df
    :param df: a DataFrame object with the columns $col_names
    :param col_names: namedtuple containing all column names
    :return: a DataFrame object with all initialized columns
    """
    fixed_sents = []
    for s_orig in df[col_names.text].values:
        s = s_orig
        try:
            s = s.encode("latin1")
        except:
            try:
                s = s.encode("utf-8")
            except:
                pass
        if type(s) != unicode:
            fixed_sents.append(unicode(s, errors="ignore"))
        else:
            fixed_sents.append(s)
    df[col_names.text] = fixed_sents
    df[col_names.tag] = map(float, df[col_names.tag])  # fix all labels to be floats
    df = prepare_prev_states_df(df, col_names)

    # if needed, prepare features
    def prepare_features_for_extractor(df, feat_extractor_cls):
        from ResearchNLP.feature_extraction import AvgGloveExtractor, BOWExtractor
        if feat_extractor_cls == AvgGloveExtractor:
            from ResearchNLP.feature_extraction.avg_glove_extractor import prepare_features_df
            df = prepare_features_df(df, col_names)
        elif feat_extractor_cls == BOWExtractor:
            df = pretokenize_df(df, col_names)
        return df

    from ResearchNLP import Constants as cn
    df = prepare_features_for_extractor(df, cn.Feature_Extractor)
    df = prepare_features_for_extractor(df, cn.Expert_FeatureExtractor)
    return df


def prepare_prev_states_df(df, col_names):
    # type: (pd.DataFrame, ColumnNames) -> pd.DataFrame
    """
    initialized prev_states in $df
    :param df: a DataFrame object with the columns $col_names
    :param col_names: namedtuple containing all column names
    :return: a DataFrame object with all the sentences with the initial history
    """
    if col_names.prev_states not in df.columns:
        df[col_names.prev_states] = None
    null_indices = pd.np.where(df[col_names.prev_states].isnull())[0]
    if len(null_indices) != 0:
        missing_prev_states = map(lambda sent: [sent], df[col_names.text][null_indices])
        prev_states_col = df[col_names.prev_states].astype(object, copy=False)
        prev_states_col[null_indices] = missing_prev_states
    return df


def get_prev_states_from_sent(df, col_names, sent):
    # type: (DataFrame, ColumnNames, basestring) -> list
    matches = df[df[col_names.text] == sent].index
    assert len(matches) >= 1, "only one match in the dataframe"  # sentences can appear several times in orig dataset
    return df[col_names.prev_states][matches[0]]


def pretokenize_df(df, col_names):
    # type: (pd.DataFrame, ColumnNames) -> pd.DataFrame
    """
    tokenizes the sentences in $df
    :param df: a DataFrame object with the columns $col_names
    :param col_names: namedtuple containing all column names
    :return: a DataFrame object with all the sentences already tokenized
    """
    if col_names.tok_text not in df.columns:
        df[col_names.tok_text] = None
    null_indices = pd.np.where(df[col_names.tok_text].isnull())[0]
    if len(null_indices) != 0:
        df.loc[null_indices, col_names.tok_text] = map(lambda sent: ' '.join(stem_tokenize(sent)),
                                                       df[col_names.text][null_indices])
    return df


def add_sent_to_df(df, col_names, new_sent):
    # type: (pd.DataFrame, ColumnNames, str) -> pd.DataFrame
    """
    adds a new sent to the end of $df and tokenizes it
    :param df: a DataFrame object with the columns $col_names
    :param col_names: namedtuple containing all column names
    :param new_sent: The new sentence we want to add
    :return: a DataFrame object with all the sentences already tokenized
    """
    pandas_util.set_cell_val(df, len(df), col_names.text, new_sent)
    return prepare_df_columns(df, col_names)


def extend_sent_history(df, col_names, curr_history):
    last_sent_history = get_prev_states_from_sent(df, col_names, curr_history[-1])
    return curr_history + last_sent_history[1:]


def add_sentences_and_histories_to_df(df, col_names, sent_history_tuples):
    # type: (pd.DataFrame, ColumnNames, list) -> pd.DataFrame
    """
    adds new sentences to the end of $df and prepare features for it
    :param df: a DataFrame object with the columns $col_names
    :param col_names: namedtuple containing all column names
    :param new_sents: List of the new setneces (basestring)
    :return: a DataFrame object with all features prepared
    """
    new_sents = map(lambda t: t[0], sent_history_tuples)
    new_histories = map(lambda t: t[1], sent_history_tuples)
    combined_df = pandas_util.append_rows_to_dataframe(df, col_names.text, new_sents)
    prev_states_col = combined_df[col_names.prev_states].astype(object, copy=False)
    prev_states_col[range(len(df), len(df) + len(sent_history_tuples))] = new_histories
    return combined_df  # prepare_df_columns(combined_df, col_names)
