from pandas import DataFrame

from ResearchNLP.classification_experts import ClassificationExpert
from ResearchNLP.classification_experts import human_expert_gui
from ResearchNLP.util_files import ColumnNames, prepare_df_columns


class HumanExpert(ClassificationExpert):
    def __init__(self, col_names, possible_tags, positive_text, negative_text):
        # type: (ColumnNames, list, str, str) -> None
        super(HumanExpert, self).__init__(col_names, possible_tags)
        self.negative_text = negative_text
        self.positive_text = positive_text

    def classify_df(self, unlabeled_df):
        # type: (DataFrame) -> DataFrame

        # use the human gui package for classifying with a human
        labeled_df = human_expert_gui.classify_by_expert(unlabeled_df, self.col_names.text,
                                                         self.col_names.tag, self.positive_text, self.negative_text)
        labeled_df[self.col_names.tag] = map(float, labeled_df[self.col_names.tag])
        labeled_df = prepare_df_columns(labeled_df, self.col_names)
        # labeled_df.to_pickle('tagged_gen_sents.pkl')  # save to file
        return labeled_df
