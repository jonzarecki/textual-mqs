import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ResearchNLP.feature_extraction.feature_extractor import FeatureExtractor
from ResearchNLP.util_files import ColumnNames


class BOWExtractor(FeatureExtractor):
    """
    implements the BOW representation for text documents
    """
    def __init__(self, instances_df, col_names):
        self.notok_vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=unicode.split,  # doesn't tokenize
            lowercase=True,
            stop_words='english',
            max_features=75
        )
        super(BOWExtractor, self).__init__(instances_df, col_names)

    def _prepare_features(self, instances_df, col_names):
        # type: (pd.DataFrame, ColumnNames) -> None
        self.init_instances = instances_df  # save the reference

        self.init_instances_features = \
            self.notok_vectorizer.fit_transform(instances_df[col_names.tok_text]).toarray()  # also prepares vectorizer for action

    def transform(self, instances_df, col_names):
        if instances_df is self.init_instances:  # speedup for the case when we just want to transform one set of insts
            return self.init_instances_features

        return self.notok_vectorizer.transform(instances_df[col_names.tok_text]).toarray()
