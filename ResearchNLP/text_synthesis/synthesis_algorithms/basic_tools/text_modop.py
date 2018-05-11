class TextModOp(object):
    """
    class representing a modification option to a given sentence
    """

    def __init__(self, orig_sent, word, word_idx, new_word):
        # type: (str, str, int, str) -> None
        self.orig_sent = orig_sent
        self.word = word
        self.word_idx = word_idx
        self.word_len = len(word)

        # conserve orig-word properties to the new-word
        if word.isupper():
            new_word = new_word.upper()
        elif word.islower():
            new_word = new_word.lower()
        elif word.capitalize() == word:
            new_word = new_word.capitalize()  # set the new-word to capitalize too
        self.new_word = new_word
        assert " " not in new_word, "switched words shouldn't have spaces in it, use _"

    def save_mem(self):
        if hasattr(self, "orig_sent"):
            del self.orig_sent
        if hasattr(self, "word"):
            del self.word
        return self

    def change_to_normal(self, orig_sent):
        self.orig_sent = orig_sent
        if not hasattr(self, "word"):
            self.word = orig_sent[self.word_idx:(self.word_idx + self.word_len)]
        else:
            self.word_len = len(self.word)
        return self

    def apply(self):
        # type: () -> basestring
        """
        apply the modification operator by switching the word from the original sentence with the new word
        :return: the new sentence
        """
        s = self.orig_sent[:self.word_idx] + \
            self.new_word + self.orig_sent[(self.word_idx + self.word_len):]

        try:
            s = s.encode("latin1")
        except:
            try:
                s = s.encode("utf-8")
            except:
                pass
        if type(s) != unicode:
            s = unicode(s, errors="ignore")

        return s

    @classmethod
    def apply_mod_ops(cls, mod_ops):
        # type: (list) -> list[basestring]
        return map(lambda m_op: m_op.apply(), mod_ops)
