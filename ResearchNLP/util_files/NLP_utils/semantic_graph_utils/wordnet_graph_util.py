class WordNetVertex:
    def __init__(self, synset):
        self.synset = synset

    def __str__(self):
        return 'wnVertex: ' + str(self.synset)

    def get_hypernyms(self):
        # type: () -> list
        """
        :return: a list of Synset, each representing a hypernym (father)
        """
        return self.synset.hypernyms()

    def get_hypernyms_degree(self, degree):
        # type: (int) -> list
        """
        returns the list of hypernyms in the requested degree (up the graph)
        :param degree: the requested degree in the graph
        :return: list of hypernyms in degree
        """
        if self.synset is None:
            return []

        if degree == 0:
            return [self.synset]

        if degree == 1:
            return self.get_hypernyms()

        hypernyms = []
        for h in self.get_hypernyms():
            hypernyms = (WordNetVertex(h).get_hypernyms_degree(degree - 1))

        return hypernyms

    def get_hyponyms_degree(self, degree):
        if self.synset is None:
            return []

        if degree == 0:
            return [self.synset]

        if degree == 1:
            return self.get_hyponyms()

        hyponyms = []
        for h in self.get_hyponyms():
            hyponyms = hyponyms + (WordNetVertex(h).get_hyponyms_degree(degree - 1))

        return hyponyms

    def get_hyponyms(self):
        # type: () -> list
        """
        :return: a list of Synset, each representing a hyponym (son)
        """
        return self.synset.hyponyms()

    def get_synset(self):
        return self.synset


def get_synset_relatives(synset, degree):
    # type: (Synset, int) -> list
    """
    expands the base word, to all it's i_th degree related words in the WordNet graph
    :param synset: a Synset, the base synset
    :param degree: the degree of relatedness we want to return
    :return: list of related word (as Synset)
    """
    # there are probably more that die before that
    hypernyms = WordNetVertex(synset).get_hypernyms_degree(degree)
    related = []
    for hn in hypernyms:
        related = related + WordNetVertex(hn).get_hyponyms_degree(degree)
    # similar_tos
    similar_tos = set([synset])
    for _ in range(degree):
        for ss in list(similar_tos):  # add all similar syn-sets to 'similar_tos'
            for similar_ss in ss.similar_tos():
                similar_tos.add(similar_ss)
    related = related + list(similar_tos)
    return related
