import cPickle as pkl

from ResearchNLP import Constants as cn
from ResearchNLP.knowledge_bases import KnowledgeBase
from ResearchNLP.util_files.NLP_utils import pos_tagger
from ResearchNLP.util_files.NLP_utils.semantic_graph_utils import get_synset_relatives
from ResearchNLP.util_files.NLP_utils.wsd import wn_disambiguate
from config import CODE_DIR


class WordNetKB(KnowledgeBase):
    def __init__(self, dist_wc):
        """
        :param dist_wc: number of words in one unit of distance
        """
        super(WordNetKB, self).__init__(dist_wc, 'wordnet_index.pkl')
        self.lemma_idx = LemmaCountIndex()

    def _load_words_from_kbase(self, context, base_word, dist):
        related_words = []
        base_sent = context.lower()
        base_synset = self._get_word_synset(base_sent, base_word)

        if base_synset is None:  # not in WordNet
            return []
        if dist == 0:
            return base_synset.lemma_names()  # synonyms

        if cn.verbose: print base_synset.__str__() + ": " + base_synset.definition()

        related_synsets = []
        for i in range(5):
            related_synsets = related_synsets + self._sort_by_appear_count(get_synset_relatives(base_synset, i))

        for word_synset in related_synsets:
            for word in word_synset.lemma_names():  # all synonyms
                if word not in related_words:
                    related_words.append(word)  # append to the end

        return related_words

    @staticmethod
    def _get_word_synset(sent, word):
        # type: (str, str) -> Synset
        """
        returns a word's synset from WordNet
        :return: the relevant Synset for $word
        """
        tagged_sent = pos_tagger.pos_tag_sent(sent)  # use the pos tagger for $pos_tag in disambiguate()
        word_tag = pos_tagger.find_word_tag(tagged_sent, word)
        word_synset = wn_disambiguate(sent, word, simp_pos_tag=word_tag)
        return word_synset

    def _sort_by_appear_count(self, synset_list):
        """
        :return: sorts the list by the number of appearances in WordNet
        """
        return sorted(synset_list, reverse=True,  # lambda returns the total number of count() in each lemma in synset
                      key=lambda synset: self.lemma_idx.get_item(synset))

    def load_knowledgebase(self):
        print "nothing to load for WordNet"

    def _extract_relevant_context(self, sent, base_word):
        return sent


# key is (word) value is the appear count in WordNet
class LemmaCountIndex(object):
    def __init__(self):
        # type: () -> None
        # contains lemma counts for each synset
        self.lemma_count_dict = pkl.load(open(CODE_DIR +
            "knowledge_bases/knowledge-base-models/wordnet_models/lemma_count_dict.pkl", 'rb'))

    def get_item(self, synset):
        str_key = synset._name
        lemma_count = self.lemma_count_dict.get(str_key)
        if lemma_count is None:
            assert False, "lemma_count_dict.pkl should contain ALL synsets"
        return lemma_count
