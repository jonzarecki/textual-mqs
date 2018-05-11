from abc import abstractmethod

import pandas as pd
from libact.base.interfaces import Labeler

from ResearchNLP.util_files import pandas_util, ColumnNames
from ResearchNLP.util_files.column_names import add_sent_to_df


class ClassificationExpert(Labeler):
    # the expert abstract base class
    def __init__(self, col_names, possible_tags):
        # type: (ColumnNames, list) -> None
        self.possible_tags = possible_tags
        self.col_names = col_names

    def classify_single_sent(self, sent):
        # type: (str) -> int
        """
        Returns the classification value of a single sentence
        """
        single_sent_df = pd.DataFrame(columns=list(self.col_names))
        single_sent_df = add_sent_to_df(single_sent_df, self.col_names, sent)

        tagged_df = self.classify_df(single_sent_df)  # classify our single sentence

        return int(pandas_util.get_cell_val(tagged_df, 0, self.col_names.tag))

    @abstractmethod
    def classify_df(self, unlabeled_df):
        # type: (pd.DataFrame) -> pd.DataFrame
        pass

    def label(self, features):
        # needed for the object being a Labeler
        assert isinstance(features, str)  # features is actually just an str
        return self.classify_single_sent(features)


def load_machine_expert(col_names, possible_tags, all_data_df):
    # type: (ColumnNames, list, pd.DataFrame) -> None
    global clsExpert
    from ResearchNLP.classification_experts import MachineExpert
    if type(clsExpert) != MachineExpert or (type(clsExpert) == MachineExpert and list(clsExpert.col_names) != list(col_names)):
        print "reload ClassificationExpert"
        clsExpert = MachineExpert(col_names, possible_tags, all_data_df)


def load_human_expert(col_names, possible_tags, pos_stmt, neg_stmt):
    # type: (ColumnNames, list, str, str) -> None
    global clsExpert
    from ResearchNLP.classification_experts import HumanExpert
    if type(clsExpert) != HumanExpert or (type(clsExpert) == HumanExpert and list(clsExpert.col_names) != list(col_names)):
        print "reload ClassificationExpert"
        clsExpert = HumanExpert(col_names, possible_tags, pos_stmt, neg_stmt)


clsExpert = ClassificationExpert  # type hint
