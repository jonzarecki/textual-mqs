import ResearchNLP.classification_experts.classification_expert as ce
from ResearchNLP import Constants as cn
from ResearchNLP.classification_experts import MachineExpert
from tests.classification_experts.test_classificationExpert import TestClassificationExpert


class TestMachineExpert(TestClassificationExpert):
    """
    tests for HumanExpert, for now only works with sentiment analysis data
    """

    @classmethod
    def setUpClass(cls):
        cn.load_codementor_sentiment_analysis_parameters()
        ce.load_machine_expert(cn.col_names, cn.relevant_tags, cn.data_df)

    def setUp(self):
        # self.data_df = cn.data_df.copy(deep=True)
        self.text_col = cn.text_col
        self.tag_col = cn.tag_col

    def test_sentiment_sanity(self):
        """
        tests simple sentences work, and that the GUI works OK
        """
        self.assertIsInstance(ce.clsExpert, MachineExpert)
        self.run_sentiment_sanity(ce.clsExpert)

