import numpy as np

from ResearchNLP.feature_extraction import FeatureExtractor


class TextExtractor(FeatureExtractor):
    """
    implements text for text documents (identity)
    """

    def _prepare_features(self, instances_df, col_names):
        self.init_instances_df = instances_df
        self.init_instances_features = np.array(instances_df[col_names.text])

    def transform(self, instances_df, col_names):
        if instances_df is self.init_instances_df:
            return self.init_instances_features  # will maybe.. saves some time
        else:  # not initial instances
            return np.array(instances_df[col_names.text])  # return the sentences in a numpy array
