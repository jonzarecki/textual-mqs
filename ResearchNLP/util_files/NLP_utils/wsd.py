# example, with cosine lesk
from pywsd.lesk import cosine_lesk as wsd_fun
from pywsd.similarity import sim


def map_simplified_tag(tag):
    """

    :param tag: the tag we want to map
    :return: a more simplified tags
    """
    if tag == 'VERB':
        return 'v'
    elif tag == 'NOUN':
        return 'n'
    elif tag == 'ADJ':
        return None  # can be both 's', and 'r' so I'll let the disambiguator decide
    return tag


def wn_disambiguate(sent, ambiguous, simp_pos_tag=None):
    """

    :param sent: the base sentence $ambiguous comes from
    :type sent: str
    :param ambiguous: an ambiguous word from $sent
    :type ambiguous: str
    :param simp_pos_tag: the pos of $ambiguous in $sent
    :return: a Synset for the ambiguous word
    :rtype: Synset
    """
    if simp_pos_tag is not None:
        simp_pos_tag = map_simplified_tag(simp_pos_tag)
    return wsd_fun(sent, ambiguous, pos=simp_pos_tag)


def word_similarity(sense1, sense2):
    """

    :type sense1: Synset
    :type sense2: Synset
    :rtype float
    :return: a number to suggest the similarity level of these 2 synsets
    """
    print "word sim"
    return sim(sense1, sense2, option='jcn')
