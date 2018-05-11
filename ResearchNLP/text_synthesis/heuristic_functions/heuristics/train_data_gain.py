from libact.models import LogisticRegression
from pandas import DataFrame, np

from ResearchNLP import Constants as cn
from ResearchNLP.text_synthesis.heuristic_functions.synthesis_state import SynState
from ResearchNLP.util_files import ColumnNames, pandas_util, ExprScores

all_heuristics = [("train-data-gain", lambda: SynStateTrainDataGain)]


class SynStateTrainDataGain(SynState):
    """
    Returns a score based on the increase of the test data accuracy,
    cheating but good for testing heuristics as it is close to the perfect heuristic
    """

    def __init__(self, state_idx, sent_df, col_names, prev_state=None):
        # type: (int, DataFrame, ColumnNames, SynStateTestDataGain) -> None

        super(SynStateTrainDataGain, self).__init__(state_idx, sent_df, col_names, prev_state)
        self.state_df = pandas_util.copy_dataframe_structure(sent_df).append(sent_df.loc[state_idx]).reset_index(drop=True)
        # self.labeled_state_df = label_df_with_expert(state_df, col_names, print_status=False)

    def get_state_score(self):
        # type: () -> float
        """ adds the state's textual state to the dataset and check the increase in accuracy"""
        if self.prev_state is None:
            return 0  # initial state score
        from ResearchNLP.z_experiments.ex_insertion_order import scores_per_add_default
        from ResearchNLP.text_synthesis.heuristic_functions.heuristics.al_heuristics import SynStateUncertainty
        ds = SynStateUncertainty.build_query_strategy(self.sent_df, self.col_names)._dataset
        clf = LogisticRegression()
        clf.train(ds)
        p0, p1 = clf.predict_proba(np.array(ds.data[self.state_idx][0].reshape(1, -1)))[0]
        labeled_df = self.sent_df[self.sent_df[self.col_names.text].notnull()]

        def kfold_gain(train_set, dev_set, state_df, col_names):
            def depth1_gain(labeled_state_df):
                ex_added_list, res_list = scores_per_add_default(labeled_state_df,
                                                                 train_set, dev_set)
                f1_list = ExprScores.list_to_f1(res_list)
                return f1_list[1] - f1_list[0]  # difference in f1 score. NOT NORMALIZED, but its supposed to be OK
            state_df.loc[0, col_names.tag] = 0
            change0 = depth1_gain(state_df)
            state_df.loc[0, col_names.tag] = 1
            change1 = depth1_gain(state_df)
            cn.add_experiment_param("5_spits_with_prob_kfold_gain")
            return p0 * change0 + p1 * change1


        # total_gain = kfold_gain(labeled_df, labeled_df, self.state_df, self.col_names)
        from sklearn.model_selection import KFold
        total_gains = []
        kf = KFold(n_splits=5)
        labeled_train_df = self.sent_df[self.sent_df[self.col_names.tag].notnull()].reset_index(drop=True)
        for train, dev in kf.split(range(len(labeled_train_df))):
            train_df = labeled_train_df.iloc[train]
            dev_df = labeled_train_df.iloc[dev]
            total_gains.append(kfold_gain(train_df, dev_df, self.state_df, self.col_names))
        inst_gain = sum(total_gains) / len(total_gains)

        return inst_gain




