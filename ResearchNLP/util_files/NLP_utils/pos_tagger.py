import spacy

from ResearchNLP.util_files import function_cache

nlp = spacy.load('en', parser=False, entity=False, matcher=False, add_vectors=False)
nlp.disable_pipes('ner')
nlp.disable_pipes('parser')


@function_cache.func_cache
def pos_tag_sent(sent, only_hash=False):
    # type: (str) -> list
    """
    returns a list of tuples for each token its (token, Part-of-Speech, st_idx)
    :param sent: the sentence we want to tokenize and tag
    :param only_hash: if set to True returns a list of ints for each POS tag (much less memory)
    """
    if not only_hash:
        ret_list = map(lambda token: (token.text, token.pos_, token.idx), nlp(sent))
    else:
        ret_list = hash(tuple(map(lambda token: token.pos, nlp(sent))))
    return ret_list


# slower than pos_tag_sent for small arrays
def pos_tag_sent_batch_only_hash(sents):
    pos_tags = map(lambda sent: function_cache.try_to_extract(pos_tag_sent, sent, only_hash=False), sents)
    not_available_tags = map(lambda doc: hash(tuple(map(lambda token: token.pos, doc))),
                             nlp.pipe(sents, batch_size=len(sents)/3+1, n_threads=3))
    count = 0
    for i in range(len(pos_tags)):
        if pos_tags[i] is not None:
            pos_tags[i] = not_available_tags[count]
            count += 1
        function_cache.force_cache_save(pos_tag_sent, pos_tags[i], sents[i], only_hash=False)
    return pos_tags


def tokenize_sent(sent):
    # type: (str) -> list
    """
    returns a list representing each word in the tokenized sentence
    """
    return map(lambda token: token.lemma_, nlp(sent, disable=['tagger']))


def find_word_tag(tagged_sent, requested_word):
    """

    :param tagged_sent:
    :param requested_word: requested word
    :return: returns the tag for the first occurrence of the $requested_word or None if there isn't any
    """
    for word, tag, _ in tagged_sent:
        if word == requested_word:
            return tag

    return None
