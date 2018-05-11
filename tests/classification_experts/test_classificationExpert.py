from unittest import TestCase

import mock
from pandas import DataFrame

from ResearchNLP.classification_experts import ClassificationExpert
from ResearchNLP.util_files import ColumnNames


class TestClassificationExpert(TestCase):
    def run_sentiment_sanity(self, clsExpert):
        """
        tests that simple sentences work
        """
        positive_sent = 'I love this move it is awesome'
        negative_sent = 'this film sucks I hate it'
        self.assertIsInstance(clsExpert, ClassificationExpert)
        self.assertEqual(1, clsExpert.label(positive_sent), "not labeled positive")
        self.assertEqual(0, clsExpert.label(negative_sent), "not labeled negative")

    @mock.patch.multiple(ClassificationExpert, __abstractmethods__=set())
    def test_single_sent_calles_label_df(self):
        """
        tests that classify_single_sent calles classify_df
        """
        col_names = ColumnNames('a', 'b', 'c', 'd', 'e', 'f')
        clsExpert = ClassificationExpert(col_names, [33, 34])
        df = DataFrame(data=[["hi", 33]], columns=['a', 'b'])
        clsExpert.classify_df = mock.MagicMock(return_value=df)

        self.assertEqual(clsExpert.classify_single_sent("hello world"), 33)
        clsExpert.classify_df.assert_called_once_with(mock.ANY)

    @mock.patch.multiple(ClassificationExpert, __abstractmethods__=set())
    def test_label_calles_classify_single_sent_df(self):
        """
        tests that label just called classify_single_sent
        """

        clsExpert = ClassificationExpert(None, [33, 34])
        clsExpert.classify_single_sent = mock.MagicMock(return_value=33)

        self.assertEqual(clsExpert.label("hello world"), 33)
        clsExpert.classify_single_sent.assert_called_once_with("hello world")


