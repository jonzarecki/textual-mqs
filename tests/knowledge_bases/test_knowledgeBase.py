from unittest import TestCase

import mock

from ResearchNLP.knowledge_bases import KnowledgeBase


class TestKnowledgeBase(TestCase):
    @mock.patch.multiple(KnowledgeBase, __abstractmethods__=set())
    def setUp(self):
        self.dist_wc = 1
        self.kb = KnowledgeBase(self.dist_wc, None)
        self.kb.load_knowledgebase = mock.MagicMock(return_value=None)
        self.kb._extract_relevant_context = mock.MagicMock(side_effect=lambda x, _: x)  # whole context

    def test_select_words_in_dist_returns_expected1(self):
        retval = self.kb.select_words_in_dist("A b C d", "A", 7)
        self.assertEqual(retval, [])

    def test_select_words_in_dist_returns_expected2(self):
        for i in range(1,6,1):
            self.kb._load_words_from_kbase = mock.Mock(return_value=['1', '2', '3', '4', '5'])

            retval = self.kb.select_words_in_dist("A b C d", "A", i)
            self.kb._load_words_from_kbase.assert_called_once_with("a b c d", "a", i)
            self.assertEqual(retval, [str(i)])

            self.kb._reset_index()

    def test_select_words_in_dist_uses_lowercase(self):
        self.kb._load_words_from_kbase = mock.MagicMock(return_value=['1', '2', '3', '4', '5'])
        self.kb.in_index = mock.Mock(return_value=False)

        retval = self.kb.select_words_in_dist("A b C d", "A", 7)
        self.assertEqual(retval, [])
        self.kb.in_index.assert_called_once_with("a", "a b c d", 7)

    def test__load_words_from_kbase_called_when_index_empty(self):
        self.kb._load_words_from_kbase = mock.Mock(return_value=['1', '2', '3', '4', '5'])
        self.kb.select_words_in_dist("A b C d", "A", 1)  # new kb, index is empty
        self.kb._load_words_from_kbase.assert_called_once_with("a b c d", "a", 1)

    def test__load_words_from_kbase_is_not_called_when_index_not_empty(self):
        self.kb.dist_wc = 1
        self.kb._load_words_from_kbase = mock.Mock(return_value=['1', '2', '3', '4', '5'])
        self.kb.select_words_in_dist("A b C d", "A", 1)  # new kb, index is empty

        self.kb._load_words_from_kbase = mock.Mock(return_value=['1', '2', '3', '4', '5'])  # new mock after the load
        for i in range(1, 6):
            self.kb.select_words_in_dist("A b C d", "A", i)
            self.kb._load_words_from_kbase.assert_not_called()

    def test__load_words_from_kbase_called_when_saved_out_of_range(self):
        self.kb._load_words_from_kbase = mock.Mock(return_value=['1', '2', '3', '4', '5'])
        self.kb.select_words_in_dist("A b C d", "A", 1)
        self.kb._load_words_from_kbase.assert_called_once_with("a b c d", "a", 1)

        self.kb._load_words_from_kbase = mock.Mock(return_value=['1', '2', '3', '4', '5', '6'])
        retval = self.kb.select_words_in_dist("A b C d", "A", 6)
        self.kb._load_words_from_kbase.assert_called_once_with("a b c d", "a", 6)
        self.assertEqual(retval, ['6'])

    def test_in_index(self):
        self.kb._load_words_from_kbase = mock.Mock(return_value=['1', '2', '3', '4', '5'])
        self.kb.select_words_in_dist("A b C d", "A", 1)
        self.kb._load_words_from_kbase.assert_called_once_with("a b c d", "a", 1)

        self.kb._load_words_from_kbase = mock.Mock(return_value=['1', '2', '3', '4', '5', '6'])
        self.kb.select_words_in_dist("A b C d", "A", 6)  # saved in index

        self.assertTrue(self.kb.in_index("A".lower(), "A b C d".lower(), 6))

    def test__extract_words_in_dist(self):
        self.assertEqual(KnowledgeBase._extract_words_in_dist(['1', '2', '3', '4', '5'], 6, 1), [])
        self.assertEqual(KnowledgeBase._extract_words_in_dist(['1', '2', '3', '4', '5'], 0, 1), [])

        self.assertEqual(KnowledgeBase._extract_words_in_dist(['1', '2', '3', '4', '5'], 1, 1), ['1'])
        self.assertEqual(KnowledgeBase._extract_words_in_dist(['1', '2', '3', '4', '5'], 3, 1), ['3'])

        self.assertEqual(KnowledgeBase._extract_words_in_dist(['1', '2', '3', '4', '5'], 2, 2), ['3', '4'])
        self.assertEqual(KnowledgeBase._extract_words_in_dist(['1', '2', '3', '4', '5'], 3, 2), ['5'])

        self.assertEqual(KnowledgeBase._extract_words_in_dist(['1', '2', '3', '4', '5'], 1, 4), ['1', '2', '3', '4'])
        self.assertEqual(KnowledgeBase._extract_words_in_dist(['1', '2', '3', '4', '5'], 2, 4), ['5'])

