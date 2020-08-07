import time
from glove import Glove

from ResearchNLP.knowledge_bases.knowledge_base import KnowledgeBase


class GloveKB(KnowledgeBase):
    def __init__(self, dist_wc, kb_foldpath, model_relpath):
        """
        :param kb_foldpath: path to the models folder
        :param model_relpath: path from the model folder to the model file
        :param dist_wc: number of words in one unit of distance
        """
        super(GloveKB, self).__init__(dist_wc, model_relpath.replace('.', '_') + "_index.pkl")
        self.model_path = kb_foldpath + model_relpath
        self.glove_model = None
        # self.load_knowledgebase()
        print "loading glove model lazily"

    def _load_words_from_kbase(self, context, base_word, dist):
        if self.glove_model is None:
            print "loading glove model, cause word: " + base_word
            self.load_knowledgebase()

        if base_word not in self.glove_model.dictionary:
            return []

        similar_wordvecs = self.glove_model.most_similar(base_word, (dist + 3) * self.dist_wc + 1)
        return [wordvec[0] for wordvec in similar_wordvecs]  # extract the actual words

    def load_knowledgebase(self):
        if self.glove_model is None:
            print "loading glove model  . . ."
            start_time = time.time()
            self.glove_model = Glove.load_stanford(self.model_path)
            print "gloveKB loading time: " + str(time.time() - start_time)

    def _extract_relevant_context(self, sent, base_word):
        return ''  # don't need any context
