import math

import numpy as np
from libact.base.dataset import Dataset
from pandas import DataFrame

from ResearchNLP.feature_extraction import FeatureExtractor
from ResearchNLP.util_files import ColumnNames, prepare_df_columns


class TextDataset(Dataset):
    """
    Specific Implementation of Dataset to work good with text datasets contained in a DataFrame
    """

    def __init__(self, df, col_names, feature_extractor, features=None):
        # type: (DataFrame, ColumnNames, FeatureExtractor, np.ndarray) -> None
        assert features is not None or feature_extractor is not None  # features=None -> extractor!=None
        # prepare data for the superclass
        self.col_names = col_names
        self.df = prepare_df_columns(df, col_names)
        self.df = df.copy(deep=True)
        self.sents = df[col_names.text]  # used in IdealLabeler
        if features is None:
            features = feature_extractor.transform(df, col_names)
        tags = map(lambda tag: tag if not math.isnan(tag) else None, df[col_names.tag].astype(float))
        def update_dataframe(entry_id, lb):
            self.df.loc[entry_id, col_names.tag] = lb
        super(TextDataset, self).__init__(features, tags)
        self.on_update(update_dataframe)

    # all other functions work fine (inherited from Dataset)

    def extract_sentence(self, id):
        # type: (int) -> str
        """
        extract the sentence with the given id (will be used to extract the sentence from make_query()
        """
        return self.sents[id]

    def extract_dataframe(self):
        # type: () -> DataFrame
        """
        extract all the information in the Dataset as a dataframe
        """
        ret_df = self.df.copy(deep=True)
        ret_df[self.col_names.tag] = map(lambda entry: int(entry[1]) if entry[1] is not None else None,
                                         self.get_entries())  # update labels
        return ret_df

    def extract_labeled_dataframe(self):
        # type: () -> DataFrame
        """
        extract all the information in the Dataset as a dataframe
        """
        ret_df = self.df.copy(deep=True)
        # return ret_df[ret_df[self.col_names.tag].notnull()].reset_index(drop=True)
        ret_df[self.col_names.tag] = map(lambda entry: int(entry[1]) if entry[1] is not None else None,
                                         self.get_entries())  # update labels
        # only labeled rows
        ret_df = ret_df[ret_df[self.col_names.tag].notnull()].reset_index(drop=True)
        return ret_df

    def update(self, entry_id, new_label):
        super(TextDataset, self).update(entry_id, new_label)
        self.df.loc[entry_id, self.col_names.tag] = new_label