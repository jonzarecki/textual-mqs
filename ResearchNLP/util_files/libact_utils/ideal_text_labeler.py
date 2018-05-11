import numpy as np
from libact.base.interfaces import Labeler

from ResearchNLP.util_files.libact_utils.text_dataset import TextDataset


class IdealTextLabeler(Labeler):
    """Ideal Text Labeler

    IdealTextLabeler is a labeler for text sentences that receives all examples the he can be queried on in advance
        and when asked for a label returns that label

    Parameters
    ----------
    text_ds: TextDataset
        Contains all the labeled data IdealTextLabeler needs

    """

    def __init__(self, text_ds):
        sents_keys = text_ds.sents
        _, tags_values = zip(*text_ds.get_entries())
        # make sure the input dataset is fully labeled
        assert (np.array(tags_values) != np.array(None)).all(), "IdealTextLabeler: all entries must be labeled"

        self.sent_dict = dict(zip(sents_keys, tags_values))  # build a sent to tag dictionary

    def label(self, text):
        # type: (basestring) -> int
        assert isinstance(text, basestring), "IdealTextLabeler: label() only accepts strings"
        return self.sent_dict[text]
