from imbal.metrics import ConfusionMatrix

def true_positive_rate(confusion_matrix : ConfusionMatrix) -> float:
    return confusion_matrix.tp() / confusion_matrix.pos()

def false_positive_rate(confusion_matrix : ConfusionMatrix) -> float:
    return confusion_matrix.fp() / confusion_matrix.neg()

def true_negative_rate(confusion_matrix : ConfusionMatrix) -> float:
    return confusion_matrix.tn() /confusion_matrix.neg()

def precision(confusion_matrix : ConfusionMatrix) -> float:
    return confusion_matrix.tp() / confusion_matrix.ppos()

def expected_tp(confusion_matrix : ConfusionMatrix) -> float:
    return confusion_matrix.pos() * confusion_matrix.ppos() / confusion_matrix.sample_size()

def expected_tn(confusion_matrix : ConfusionMatrix) -> float:
    return confusion_matrix.neg() * confusion_matrix.pneg() / confusion_matrix.sample_size()

def expected_correct(confusion_matrix : ConfusionMatrix) -> float:
    return expected_tp(confusion_matrix) + expected_tn(confusion_matrix)