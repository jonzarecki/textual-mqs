import pandas as pd

from ResearchNLP import Constants as cn
from ResearchNLP.util_files import ColumnNames, prepare_df_columns, pandas_util
from ResearchNLP.classification_experts.classification_expert import ClassificationExpert


class MachineExpert(ClassificationExpert):
    def __init__(self, col_names, possible_tags, all_data_df):
        # type: (ColumnNames, list, pd.DataFrame) -> None
        super(MachineExpert, self).__init__(col_names, possible_tags)
        self.all_data_df = all_data_df
        self.col_names = col_names

    def classify_df(self, unlabeled_df):
        # type: (pd.DataFrame) -> pd.DataFrame
        # if cn.balance_dataset:
        #     self.all_data_df = pandas_util.imbalance_dataset(cn.data_df, cn.tag_col, 0.5,
        #                                                      cn.pos_tags, cn.neg_tags)

        # build the feature extractor
        combined_df = pd.concat([self.all_data_df, unlabeled_df])
        extractor_cls = cn.Expert_PredictionModel.get_FeatureExtractor_cls()
        if extractor_cls is None:
            extractor_cls = cn.Expert_FeatureExtractor

        extractor = extractor_cls(combined_df, self.col_names)
        X_all = extractor.transform(combined_df, self.col_names)  # use pre-extracted features

        # extract all the features
        X_unlabeled = X_all[len(self.all_data_df[self.col_names.text]):]
        X_train = X_all[:len(self.all_data_df[self.col_names.text])]
        y_train = self.all_data_df[self.col_names.tag].tolist()

        model = cn.Expert_PredictionModel(X_train, y_train)  # expert model trains on all data (makes it an expert)

        y_pred = model.train_model_and_predict(X_unlabeled)
        # print y_pred.tolist()

        labeled_df = unlabeled_df.copy(deep=True)
        labeled_df[self.col_names.tag] = map(float, y_pred)  # add predictions to the df (simulates a human label)
        labeled_df = prepare_df_columns(labeled_df, self.col_names)

        return labeled_df  # return the now tagged df
