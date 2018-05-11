from sklearn.linear_model import LogisticRegression

from ResearchNLP.prediction_models.prediction_model import PredictionModel


class LinearModel(PredictionModel):
    """
    A PredictionModel class, implemented to use sklearn's linear LogisticRegression
    """

    # inherits __init__()

    def _train(self, X_unlabeled):
        # extract the feature vector for the new linear model

        misc = {"X_unlabeled": X_unlabeled}  # will contain data needed by the _predict function
        log_model = LogisticRegression()  # uses cn.Feature_Extractor features
        log_model = log_model.fit(X=self.X_train, y=self.y_train)  # model is now trained

        return log_model, misc

    def _predict(self, model, misc):
        return model.predict(misc["X_unlabeled"])
