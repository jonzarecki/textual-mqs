import ResearchNLP.classification_experts.classification_expert as ce
from ResearchNLP import Constants as cn
from ResearchNLP.Constants import load_codementor_sentiment_analysis_parameters
from ResearchNLP.classification_experts import HumanExpert
from test_classificationExpert import TestClassificationExpert


class TestHumanExpert(TestClassificationExpert):
    """
    tests for HumanExpert, for now only works with sentiment analysis data
    """

    @classmethod
    def setUpClass(cls):
        load_codementor_sentiment_analysis_parameters()
        ce.load_human_expert(cn.col_names, cn.relevant_tags, cn.pos_statement, cn.neg_statement)

    def setUp(self):
        # self.data_df = cn.data_df.copy(deep=True)
        self.text_col = cn.text_col
        self.tag_col = cn.tag_col

    def test_sentiment_sanity(self):
        """
        tests simple sentences work, and that the GUI works OK
        """
        self.skipTest(reason="requires human labor")
        self.assertIsInstance(ce.clsExpert, HumanExpert)
        self.run_sentiment_sanity(ce.clsExpert)

