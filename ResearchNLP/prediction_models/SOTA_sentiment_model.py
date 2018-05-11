import numpy as np

from ResearchNLP.feature_extraction import TextExtractor
from ResearchNLP.prediction_models.SOTA_sentiment_library.grds_helper import predict_sentiment_bulk
from ResearchNLP.prediction_models.prediction_model import PredictionModel


class SOTASentimentModel(PredictionModel):
    """
    A PredictionModel class, implemented to use openAI's
     state-of-the-art sentiment analysis tool
    """

    @staticmethod
    def get_FeatureExtractor_cls():
        return TextExtractor

    def _train(self, X_unlabeled):
        # extract the feature vector for the new linear model
        assert isinstance(X_unlabeled[0], basestring), "SOTASentimentModel accepts only text features"
        unlabeled_sents = X_unlabeled.tolist()  # X
        return None, {"unlabeled_sents": unlabeled_sents}

    def _predict(self, model, misc):
        sents = misc["unlabeled_sents"]
        return np.asarray(predict_sentiment_bulk(sents))
