from unittest import TestCase

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from ResearchNLP.feature_extraction import BOWExtractor
from ResearchNLP import Constants as cn
from ResearchNLP.util_files.NLP_utils.stemmer import stem_tokenize
from ResearchNLP.util_files.column_names import pretokenize_df


class TestBOWExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cn.load_codementor_sentiment_analysis_parameters(d_measure=10)
        cn.data_df = pretokenize_df(cn.data_df, cn.col_names)

    def setUp(self):
        self.orig_vect = CountVectorizer(
            analyzer='word',
            tokenizer=stem_tokenize,
            lowercase=True,
            stop_words='english',
            max_features=75
        )
        self.data_df = cn.data_df
        self.text_col = cn.text_col
        self.col_names = cn.col_names

    def test__prepare_features(self):
        """
        tests if both normal vectorizer and orignal one (in stemmer) have the same vocab
        relient on that BOW has a vectorizer, weak.
        """
        ext = BOWExtractor(self.data_df, self.col_names
                           )  # calls _prepare_features
        self.orig_vect.fit(self.data_df[self.text_col])
        self.assertTrue(ext.notok_vectorizer.get_feature_names() == self.orig_vect.get_feature_names())

    def test_transform(self):
        """
        tests if the 2 vectorizers create the same features for the data
        """
        sents = self.data_df[self.text_col].values
        ext = BOWExtractor(self.data_df, self.col_names)  # calls _prepare_features
        self.orig_vect.fit(self.data_df[self.text_col])
        # do they have the same features
        self.assertTrue(np.array_equiv(ext.transform(self.data_df, self.col_names),
                                       self.orig_vect.transform(sents).toarray()))

    def test_tokenizing_before_doesnt_change_features(self):
        """
        tests if tokenizing the sentences beforehand changes the output
        """
        # self.skipTest("Doesn't work for some reason :(")
        sents = self.data_df[self.text_col].values
        notok_vect = CountVectorizer(
            analyzer='word',
            tokenizer=unicode.split,  # doesn't tokenize
            lowercase=True,
            stop_words='english',
            max_features=75
        )

        tokenized_sents = map(lambda sent: ' '.join(stem_tokenize(sent)), sents)

        notok_vect.fit(tokenized_sents)
        self.orig_vect.fit(sents)
        # do they have the same features
        # normal extractor with no tokenize extractor

        self.assertTrue(not set(self.orig_vect.get_feature_names()).difference(set(notok_vect.get_feature_names())))
        self.assertTrue(not set(notok_vect.get_feature_names()).difference(set(self.orig_vect.get_feature_names())))

        for i in range(len(sents)):
            self.assertEquals(notok_vect.transform([tokenized_sents[i]]).sum(),
                              self.orig_vect.transform([sents[i]]).sum())

    def test_tokenizing_before_doesnt_change_features2(self):
        """
        tests if tokenizing the sentences beforehand changes the output
        """
        sents = self.data_df[self.text_col].values
        tok_ext = BOWExtractor(self.data_df, self.col_names)
        tokenized_sents = map(lambda sent: ' '.join(stem_tokenize(sent)), sents)
        notok_ext = BOWExtractor(self.data_df, self.col_names)

        for i in range(len(sents)):
            self.assertEquals(notok_ext.notok_vectorizer.transform([tokenized_sents[i]]).sum(),
                              tok_ext.notok_vectorizer.transform([tokenized_sents[i]]).sum())

        self.assertEqual(tok_ext.transform(self.data_df, self.col_names).sum(),
                         notok_ext.transform(self.data_df, self.col_names).sum())
