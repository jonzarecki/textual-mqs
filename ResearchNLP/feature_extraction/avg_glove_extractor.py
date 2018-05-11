import os
import time

import numpy as np
import pandas as pd
from glove import Glove

from ResearchNLP.feature_extraction.feature_extractor import FeatureExtractor
from ResearchNLP.util_files.parallel_load import ParallelLoad
from config import CODE_DIR


class AvgGloveExtractor(FeatureExtractor):
    """
    implements an average glove vector representation for text documents
    """
    def __init__(self, instances_df, col_names):
        super(AvgGloveExtractor, self).__init__(instances_df, col_names)

    def _prepare_features(self, instances_df, col_names):
        pass

    def transform(self, instances_df, col_names):
        assert len(pd.np.where(instances_df[col_names.feature_repr].isnull())[0]) == 0, "all rows should have features"
        return np.array(list(instances_df[col_names.feature_repr]))  # extract features


def initialize_encoder():
    model_path = os.path.join(CODE_DIR, "knowledge_bases/knowledge-base-models/glove_models/glove.6B.300d.txt")
    print "loading glove encoder ..."
    start_time = time.time()
    glove_model = Glove.load_stanford(model_path)
    print "AvgGloveExtractor loading time: " + str(time.time() - start_time)
    return glove_model


glove_model = ParallelLoad(loading_fun=initialize_encoder)
vocab_set = None


def all_in_vocab(sent):
    initialize_vocab()
    # import ResearchNLP.Constants as cn
    # cn.add_experiment_param('glove_token_split')
    # return " ".join(filter(lambda word: word in vocab_set, pos_tagger.tokenize_sent(sent)))
    return " ".join(filter(lambda word: word in vocab_set, sent.lower().split()))


def initialize_vocab():
    global vocab_set
    if vocab_set is None:
        vocab_set = set(glove_model.obj.dictionary.iterkeys())


total_diff = 0
diff_count = 0


def get_sentence_representation(sent):
    import ResearchNLP.Constants as cn
    cn.add_experiment_param('glove300')
    reduced_sent = all_in_vocab(sent).split()
    if len(reduced_sent) == 0:
        # print "reduced sent: " + str(sent)
        return [0.0] * len(glove_model.obj.word_vectors[0])  # zeros representation
    global total_diff, diff_count
    total_diff += len(sent.split()) - len(reduced_sent)
    diff_count += 1
    return sum(map(lambda word: glove_model.obj.word_vectors[glove_model.obj.dictionary[word]].__array__(), reduced_sent)) \
             / len(reduced_sent)


def prepare_features_df(df, col_names):
    # type: (pd.DataFramae, ColumnNames) -> pd.DataFrame
    """
    extracts features for the sentences in $df
    :param df: a DataFrame object with the columns $col_names
    :param col_names: namedtuple containing all column names
    :return: a DataFrame object with all the sentences with their feature representation
    """
    if col_names.feature_repr not in df.columns:
        df[col_names.feature_repr] = None
    null_indices = pd.np.where(df[col_names.feature_repr].isnull())[0]
    if len(null_indices) != 0:
        from ResearchNLP.knowledge_bases import kb_helper
        if kb_helper.kb_type != 'GloVe':
            # clean underlines (left from WordNet, w2v, etc.)
            features = map(lambda sent: get_sentence_representation(sent.replace("_", " ")), df[col_names.text][null_indices])
        else:
            features = map(lambda sent: get_sentence_representation(sent), df[col_names.text][null_indices])

        df[col_names.feature_repr] = df[col_names.feature_repr].astype(object)
        if len(null_indices) == len(df):
            df[col_names.feature_repr] = features
        else:
            feature_repr_col = df[col_names.feature_repr].astype(object, copy=False)
            feature_repr_col.loc[null_indices] = features
            # for order, null_idx in enumerate(null_indices):
            #     feature_repr_col[null_idx] = features[order]
    return df
