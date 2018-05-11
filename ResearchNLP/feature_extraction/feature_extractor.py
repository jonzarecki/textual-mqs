from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd
from six import with_metaclass

from ResearchNLP.util_files import ColumnNames


class FeatureExtractor(with_metaclass(ABCMeta, object)):
    """
    abstract class used to define an interface for feature extraction,
    will be used by the MachineExperts to decide what to send to their models (which USUALLY expect numeric features)

    !! implementing classes should only implement __prepare_features and transform, not __init__ !!
    """

    def __init__(self, instances_df, col_names):
        # type: (pd.DataFrame, ColumnNames) -> None
        """
        calls _prepare_features
        """
        self._prepare_features(instances_df, col_names)

    @abstractmethod
    def _prepare_features(self, instances_df, col_names):
        # type: (pd.DataFrame, ColumnNames) -> None
        """
        prepares the feature extractor object to transform new texts to feature vectors
        :param instances_df: a dataframe that contains all the instances we want to be taken into account
                                when building the feature extractor
        :param col_names: column names of the dataframe
        :return: None
        """
        pass

    @abstractmethod
    def transform(self, instances_df, col_names):
        # type: (pd.DataFrame, ColumnNames) -> np.ndarray
        """
        transform the new instances to their feature representation (as a numpy matrix)
        :param instances_df: a dataframe that contains all the instances we want to turn into features
        :param col_names: column names of the dataframe
        :return: a numpy matrix, containing the defined feature vectors (each instance is represented in a row, same order)
        """
        pass
