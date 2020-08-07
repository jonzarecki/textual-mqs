from unittest import TestCase
from ResearchNLP.util_files.NLP_utils import pos_tagger


class TestPos_tag_sent(TestCase):
    def test_pos_tag_sent(self):
        """
        checks pos tag sent does what it should, a little dependent on the tokenization
        so it can fail possibly in the future
        """
        sent = u"the quick,   brown. ,  Fox. jumps, over'' the lazy\" dog"
        tagged_sent = pos_tagger.pos_tag_sent(sent)
        correct_locations = [0, 4, 9, 11, 13, 18, 20, 22, 23, 26, 28, 33, 35, 39, 42, 46, 50, 52]
        correct_tokens = [u'the',
                          u'quick',
                          u',',
                          u'  ',
                          u'brown',
                          u'.',
                          u',',
                          u' ',
                          u'Fox',
                          u'.',
                          u'jumps',
                          u',',
                          u'over',
                          u"''",
                          u'the',
                          u'lazy',
                          u'"',
                          u'dog']
        self.assertEqual(map(lambda x: x[0], tagged_sent), correct_tokens)
        self.assertEqual(map(lambda x: x[2], tagged_sent), correct_locations)
