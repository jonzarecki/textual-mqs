from libact.base.interfaces import Labeler
from six.moves import input


class InteractiveTextLabeler(Labeler):
    """Interactive Text Labeler

    InteractiveTextLabeler is a Labeler object that shows the text sentence in the command line,
    then waits for the oracle to label it

    Parameters
    ----------
    label_name: list
        Let the label space be from 0 to len(label_name)-1, this list
        corresponds to each label's name.

    """

    def __init__(self, **kwargs):
        self.label_name = kwargs.pop('label_name', None)

    def label(self, text):

        banner = "Enter the associated label with the text: "

        if self.label_name is not None:
            banner += '['
            for i, label in enumerate(self.label_name):
                banner += str(i) + ":" + label + ", "
            banner = banner[:-2] + ']'

        print  # newline
        print text
        lbl = int(input(banner))

        while lbl not in range(len(self.label_name)):
            print('Invalid label, please re-enter the associated label.')
            lbl = int(input(banner))

        return lbl
