import nltk
from nltk.stem.porter import PorterStemmer

#######
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = PorterStemmer()


def stem_tokens(tokens, stmr):
    stemmed = []
    for item in tokens:
        stemmed.append(stmr.stem(item))
    return stemmed


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = nltk.WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]


def tokenize(text):
    # remove non letters (doesn't really need to remove non-letters)
    # text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    return nltk.word_tokenize(text)


def stem_tokenize(text):
    text = text.lower()
    tokens = tokenize(text)
    stems = stem_tokens(tokens, stemmer)  # stem
    return stems
