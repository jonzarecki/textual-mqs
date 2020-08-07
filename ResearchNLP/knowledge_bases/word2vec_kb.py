import time
from gensim.models.word2vec import Word2Vec

from ResearchNLP.knowledge_bases import KnowledgeBase


class Word2VecKB(KnowledgeBase):
    def __init__(self, dist_wc, kb_foldpath, model_relpath):
        """
        :param dist_wc: number of words in one unit of distance
        :param kb_foldpath: path to the models folder
        :param model_relpath: path from the model folder to the model file
        """
        super(Word2VecKB, self).__init__(dist_wc, model_relpath.replace('.', '_') + "_index.pkl")
        self.model_path = kb_foldpath + model_relpath
        print "loading word2vec model lazily"
        self.w2v_model = None
        # self.w2v_model = Word2Vec.load_word2vec_format(model_path, binary=model_binary)

    def _load_words_from_kbase(self, context, base_word, dist):
        if self.w2v_model is None:
            print "loading w2v model, cause word: " + base_word
            self.load_knowledgebase()

        if base_word not in self.w2v_model.vocab:
            return []

        similar_wordvecs = self.w2v_model.most_similar([base_word], [], (dist + 3) * self.dist_wc)  # up to dist
        return [wordvec[0] for wordvec in similar_wordvecs]  # extract the actual words

    def load_knowledgebase(self):
        if self.w2v_model is None:
            print "loading w2v model  . . ."
            start_time = time.time()
            # self.w2v_model = Word2Vec.load_word2vec_format(self.model_path, binary=self.model_binary)
            self.w2v_model = Word2Vec.load(self.model_path, mmap='r')
            # self.w2v_model.init_sims(replace=True)
            print "w2vKB loading time: " + str(time.time() - start_time)

    def _extract_relevant_context(self, sent, base_word):
        return ''  # don't need any context
