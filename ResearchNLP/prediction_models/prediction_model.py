from abc import abstractmethod, ABCMeta

from numpy import ndarray
from six import with_metaclass

from ResearchNLP.feature_extraction import FeatureExtractor


class PredictionModel(with_metaclass(ABCMeta, object)):
    """
    Abstract Prediction Class,
        Receives the examples in the feature space (not sentences)
        implementing classes will have to implement the _train and _predict functions
            each with its own unique model class.
        The design tries and extract only the model unique code into the implementing classes
    """

    def __init__(self, X_train, y_train):
        # type: (ndarray, list) -> None
        self.X_train = X_train
        self.y_train = y_train

    def train_model_and_predict(self, X_unlabeled):
        # type: (ndarray) -> ndarray
        """
        trains on training data (with features also from $unlabeled_data_df) and classifies them
        :param X_unlabeled: ndarray containing the new unlabeled example's features
        :return: y_pred, the models prediction on each instance in $unlabeled_data_df
        """
        model, misc = self._train(X_unlabeled)
        y_pred = self._predict(model, misc)

        return y_pred

    # can be overridden by each model chooses it's representation
    # static methods are inherited in python
    @staticmethod
    def get_FeatureExtractor_cls():
        # type: () -> Type[FeatureExtractor]
        return None


    @abstractmethod
    def _train(self, X_unlabeled):
        raise NotImplementedError

    @abstractmethod
    def _predict(self, model, misc):
        raise NotImplementedError

