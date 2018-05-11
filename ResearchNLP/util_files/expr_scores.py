
class ExprScores(object):
    # helper class to keep the scores from each experiment
    f1_name = "f1"
    roc_auc_name = "roc auc"
    acc_name = "accuracy"

    def __init__(self, f1, roc_auc, acc):
        # type: (float, float, float) -> None
        self.f1 = f1
        self.roc_auc = roc_auc
        self.acc = acc

    def __iter__(self):
        return [self.f1, self.roc_auc, self.acc].__iter__()

    def __add__(self, other):
        # type: (ExprScores) -> ExprScores
        return self.__radd__(other)

    def __radd__(self, other):
        # type: (ExprScores) -> ExprScores
        if type(other) == int:  # for sum()
            return self
        return ExprScores(self.f1 + other.f1, self.roc_auc + other.roc_auc, self.acc + other.acc)

    def __div__(self, denom):
        assert type(denom) == float, "ExprScores only divs with floats"
        return ExprScores(self.f1 / denom, self.roc_auc / denom, self.acc / denom)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)

    @staticmethod
    def enumerate_score_types():
        return [(ExprScores.f1_name, ExprScores.list_to_f1), (ExprScores.roc_auc_name, ExprScores.list_to_roc_auc),
                (ExprScores.acc_name, ExprScores.list_to_acc)]

    @staticmethod
    def list_to_f1(lst):
        return map(lambda expr_scr: expr_scr.f1, lst)

    @staticmethod
    def list_to_roc_auc(lst):
        return map(lambda expr_scr: expr_scr.roc_auc, lst)

    @staticmethod
    def list_to_acc(lst):
        return map(lambda expr_scr: expr_scr.acc, lst)

    @staticmethod
    def expr_src_list_to_seperate_lists(lst):
        return [ExprScores.list_to_f1(lst), ExprScores.list_to_roc_auc(lst), ExprScores.list_to_acc(lst)]
