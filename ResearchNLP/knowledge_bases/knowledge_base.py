import cPickle
import os
from abc import ABCMeta, abstractmethod

from six import with_metaclass

from ResearchNLP import Constants as cn
from ResearchNLP.util_files.NLP_utils.pos_tagger import tokenize_sent
from config import CODE_DIR

index_folder = CODE_DIR + 'indexes/knowledge-base-models/'


# the Knowledge-base abstract base class
class KnowledgeBase(with_metaclass(ABCMeta, object)):
    def __init__(self, dist_wc, index_relpath):
        # type: (int, str) -> None
        """
        :param index_relpath: the relative path in which the index is saved for the particular model
        :param dist_wc: number of words in one unit of distance
        """
        self.dist_wc = dist_wc
        self.index_relpath = index_relpath
        if index_relpath is not None:  # load index from file
            index_path = index_folder + index_relpath
            if os.path.isfile(index_path):
                try:
                    class Object():
                        pass
                    self.sim_words_index = Object()
                    self.sim_words_index.obj = SimilarWordsIndex()
                    # self.sim_words_index = ParallelLoad(pkl_path=index_path)
                    return  # loaded index from file
                except ImportError:
                    print "Pickle Import Error"  # probably changed some file names

        # load failed, start new index
        assert not hasattr(self, 'sim_words_index')  # assert we didn't load anything

        class Object():
            pass
        self.sim_words_index = Object()
        self.sim_words_index.obj = SimilarWordsIndex()

    def select_words_in_dist(self, sent, base_word, dist):
        # type: (str, str, int) -> list
        """
        returns words with semantic_dist($base_word,w')=$dist
        :param sent: the sentence we took $base_word from
        :param base_word: the word we want to find similar words to
        :param dist: the distance measure we want the returned words to be in.
        :return: list of word in the requested semantic distance
        """
        sent = sent.lower()
        context = self._extract_relevant_context(sent, base_word)
        # base_word = base_word.lower()  # stem the word ? won't work because the kb won't find the stemmed word

        if self.sim_words_index.obj.is_not_in_kb(base_word):
            if cn.verbose: print "'" + base_word + "' is not in the knowledge-base"
            return []

        if self.in_index(base_word, context, dist):  # is saved in index
            related_words = self.sim_words_index.obj.get_item(base_word, context)[0]
        else:
            if cn.verbose: print "loading word from knowledge-base"
            related_words = self._load_words_from_kbase(context, base_word, dist)
            base_word_stem = tokenize_sent(base_word)[0]
            related_words = filter(lambda w: tokenize_sent(w)[0] != base_word_stem, related_words)  # kick words
            if not related_words:  # no related words
                if cn.verbose: print "'" + base_word + "' is not in the knowledge-base"
                self.sim_words_index.obj.save_item_not_in_kb(base_word)
                return []
            if base_word in related_words:
                related_words.remove(base_word)
            related_words = filter(lambda word: word != base_word, related_words)
            self.sim_words_index.obj.add_item(base_word, context, related_words, len(related_words)/self.dist_wc)

        return self._extract_words_in_dist(related_words, dist, self.dist_wc)

    @abstractmethod
    def _load_words_from_kbase(self, context, base_word, dist):
        # type: (str, str, int) -> list
        """
        returns the words with semantic_dist($base_word,w')=$dist from the knowledge-base
            without using the index
        :param context: the sentence we took $base_word from
        :param base_word: the word we want to find similar words to
        :param dist: the distance measure we want the returned words to be in.
        :return: list of word up to the semantic distance
        """
        pass

    @abstractmethod
    def load_knowledgebase(self):
        """
        explicitly load the knowledge-base
                saves us from loading it on each thread or problems like that
        """
        pass

    @abstractmethod
    def _extract_relevant_context(self, sent, base_word):
        # type: (str, str) -> str
        """
        returns a function which extracts for each sentence its relevant context.
            WordNet might need the entire sent, but w2v currently only uses the current word
            :param sent:
            :param base_word:
        """
        pass

    def save_index_to_file(self):
        if index_folder is not None:
            with open(index_folder + self.index_relpath, "wb") as f:
                cPickle.dump(self.sim_words_index.obj, f)

    def in_index(self, word, sent=None, dist=1):
        # type: (str, str, int) -> bool
        """

        :param word: the word we want to get the index
        :param sent: optional parameter, use if you want only index saved with the same sent
        :param dist: the distance the user want to extract from the index
        :return: bool
        """
        context = self._extract_relevant_context(sent, word)
        if (word, context) in self.sim_words_index.obj:
            return self.sim_words_index.obj.get_item(word, context)[1] >= dist
        return False

    def _reset_index(self):
        """
        resets the index, mainly used for testing
        :return:
        """
        self.sim_words_index.obj = SimilarWordsIndex()

    @staticmethod
    def _extract_words_in_dist(word_list, dist, dist_wc):
        # type: (list, int, int) -> list
        """
        extracts only the words in the wanted distance from word_list
        :param word_list: the list of words from most-similar to least-similar
        :param dist: the distance we want to extract
        :param dist_wc: the number of words in each distance measure
        :return: list of words in the desired distance
        """
        return word_list[(dist - 1) * dist_wc: dist * dist_wc] if dist != 0 else word_list[:dist_wc]


# key is (word, sent) value is the list of similar words
class SimilarWordsIndex(object):
    def __init__(self):
        # type: () -> None
        self.sim_words_dict = {}  # will contain for each words the it's most similar words
        self.not_in_kb = set()  # will contain the words that we know aren't in the knowledge-base

    def __contains__(self, (word, context)):
        # type: (str, str) -> bool
        """

        :param word: the word we want to get the index
        :param context: optional parameter, if None ignore the sent
        :return: bool
        """
        # we know the word isn't in the knowledge-base OR didn't see the word yet
        if word in self.not_in_kb or word not in self.sim_words_dict:
            return False
        if context is None:  # don't care about sent
            return True  # we found the word is in the index

        return context in self.sim_words_dict[word]  # use primary key, and then search with secondary key

    def save_item_not_in_kb(self, word):
        """
        saves that the word is not in the knowledge-base, saves us to open it up and look for it
        """
        self.not_in_kb.add(word)

    def is_not_in_kb(self, word):
        return word in self.not_in_kb

    def get_item(self, word, context):
        assert self.__contains__((word, context)), "function assumes (word,context) is saved in the index"

        if context is not None:
            return self.sim_words_dict[word][context]
        else:  # don't care about sent
            # return value with the largest up_to_dist
            return max(self.sim_words_dict[word].itervalues(), key=lambda _, up_to_dist: up_to_dist)

    def add_item(self, word, context, word_list, up_to_dist):
        # type: (str, str, list, int) -> None
        """
        add the parameters to the index for future use
        :param word_list: the list of words we want to add to the index
        :param word: the base word
        :param context: the context $word was extracted from
        :param word_list: all similar words up to a distance
        :param up_to_dist: the distance the word list is up-to
        :return: None
        """
        if word not in self.sim_words_dict:
            self.sim_words_dict[word] = {context: (word_list, up_to_dist)}  # create new sub-dict
        else:
            self.sim_words_dict[word][context] = (word_list, up_to_dist)  # add to existing sub-dict
