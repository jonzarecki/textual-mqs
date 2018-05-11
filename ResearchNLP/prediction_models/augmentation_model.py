import numpy as np

from ResearchNLP.prediction_models.prediction_model import PredictionModel


class AugmentationModel(PredictionModel):
    """
    A PredictionModel class, implemented to keep the class the same as it's origin
    """

    # inherits __init__()

    @staticmethod
    def get_FeatureExtractor_cls():
        from ResearchNLP.feature_extraction import FeatureExtractor

        class OriginExtractor(FeatureExtractor):
            def _prepare_features(self, instances_df, col_names):
                pass

            def transform(self, instances_df, col_names):
                return np.array(map(lambda (i, row): row[col_names.prev_states][-1], instances_df.iterrows()))
        return OriginExtractor

    def _train(self, X_unlabeled):
        # extract the feature vector for the new linear model
        assert isinstance(X_unlabeled[0], basestring), "AugmentationModel accepts only text features"
        inst_origins = X_unlabeled.tolist()  # X
        return None, {"inst_origins": inst_origins}

    def _predict(self, model, misc):
        inst_origins = misc["inst_origins"]
        from ResearchNLP import Constants as cn
        orig_sents_list = cn.data_df[cn.col_names.text].tolist()
        orig_sent_tags = map(lambda orig: float(cn.data_df[cn.col_names.tag][orig_sents_list.index(orig)]),
                             inst_origins)  # find original tags
        return np.asarray(orig_sent_tags)