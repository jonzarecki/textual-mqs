from ResearchNLP import Constants as cn
from ResearchNLP.knowledge_bases import KnowledgeBase, GloveKB, Word2VecKB, WordNetKB
from config import CODE_DIR

k_base = KnowledgeBase  # type hint
knowledge_bases_foldpath = CODE_DIR + "knowledge_bases/knowledge-base-models/"
kb_type = 'NO_KB'
old_max_procs = None


def save_previous_model():
    global k_base, kb_type
    if isinstance(k_base, KnowledgeBase):  # concrete model behind 'k_base'
        k_base.save_index_to_file()


def load_word2vec_model(relpath='word2vec_models/GoogleNews-vectors-negative300_trimmed.bin'):
    global k_base, kb_type
    kb_type = 'w2v'
    save_previous_model()
    cn.add_experiment_param(kb_type)
    k_base = Word2VecKB(cn.distance_measure, knowledge_bases_foldpath, relpath)


def load_dep_word2vec_model(relpath='word2vec_models/deps_trimmed.words'):
    global k_base, kb_type
    kb_type = 'dep_w2v'
    save_previous_model()
    cn.add_experiment_param(kb_type)
    k_base = Word2VecKB(cn.distance_measure, knowledge_bases_foldpath, relpath)


def load_WordNet_model():
    global k_base, kb_type
    kb_type = 'WordNet'
    save_previous_model()
    cn.add_experiment_param(kb_type)
    k_base = WordNetKB(cn.distance_measure)


def load_GloVe_model(relpath='glove_models/glove.6B.100d.txt'):
    global k_base, kb_type
    kb_type = 'GloVe'
    save_previous_model()
    cn.add_experiment_param(kb_type)
    k_base = GloveKB(cn.distance_measure, knowledge_bases_foldpath, relpath)


if __name__ == '__main__':
    cn.verbose = True
    cn.distance_measure = 10
    glove_kb = GloveKB(cn.distance_measure, knowledge_bases_foldpath, 'glove_models/glove.6B.100d.txt')
    glove_kb._reset_index()
    glove_kb.load_knowledgebase()
    wn_kb = WordNetKB(cn.distance_measure)
    wn_kb._reset_index()
    w2v_kb = Word2VecKB(cn.distance_measure, knowledge_bases_foldpath, 'word2vec_models/GoogleNews-vectors-negative300_trimmed.bin')
    w2v_kb._reset_index()
    w2v_kb.load_knowledgebase()
    dep_w2v_kb = Word2VecKB(cn.distance_measure, knowledge_bases_foldpath, 'word2vec_models/deps_trimmed.words')
    dep_w2v_kb._reset_index()
    dep_w2v_kb.load_knowledgebase()
    # k_base.select_words_in_dist('Dumbeldore is my favorite character', 'favorite', 1)
    a=1

