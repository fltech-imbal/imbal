from .true_positive_rate import TruePositiveRate
from .false_positive_rate import FalsePositiveRate
from .true_negative_rate import TrueNegativeRate
from .expected_true_positive import ExpectedTruePositive
from .expected_true_negative import ExpectedTrueNegative
from .expected_correct import ExpectedCorrect
from .true_skill_statistic import TrueSkillStatistic
from .j_statistic import JStatistic
from .youdens_index import YoudensIndex
from .heikde_skill_score import HeikdeSkillScore
from .gilbert_skill_score import GilbertSkillScore
from .critical_success_index import CriticalSuccessIndex
from .bounded_auc import BoundedAUC
from .confusion_matrix import ConfusionMatrix, ConfusionMatrixData

from keras.src.metrics import ALL_OBJECTS_DICT
from keras.src.utils.naming import to_snake_case

IMBAL_OBJECTS = {
    TruePositiveRate,
    FalsePositiveRate,
    TrueNegativeRate,
    ExpectedTruePositive,
    ExpectedTrueNegative,
    ExpectedCorrect,
    TrueSkillStatistic,
    HeikdeSkillScore,
    GilbertSkillScore,
    CriticalSuccessIndex,
    BoundedAUC
}

ALL_OBJECTS_DICT.update({cls.__name__: cls for cls in IMBAL_OBJECTS})
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in IMBAL_OBJECTS}
)
ALL_OBJECTS_DICT.update(
    {
        'tpr' : TruePositiveRate,
        'TPR' : TruePositiveRate,
        'fpr' : FalsePositiveRate,
        'FPR' : FalsePositiveRate,
        'tnr' : TrueNegativeRate,
        'TNR' : TrueNegativeRate,
        'expected_tp' : ExpectedTruePositive,
        'expected_TP' : ExpectedTruePositive,
        'chance_hit' : ExpectedTruePositive,
        'expected_tn' : ExpectedTrueNegative,
        'expected_TN' : ExpectedTrueNegative,
        'ec' : ExpectedCorrect,
        'EC' : ExpectedCorrect,
        'tss' : TrueSkillStatistic,
        'TSS' : TrueSkillStatistic,
        'j_statistic' : TrueSkillStatistic,
        'youdens_index' : TrueSkillStatistic,
        'hss' : HeikdeSkillScore,
        'HSS' : HeikdeSkillScore,
        'gss' : GilbertSkillScore,
        'GSS' : GilbertSkillScore,
        'gs' : GilbertSkillScore,
        'GS' : GilbertSkillScore,
        'threat_score' : CriticalSuccessIndex,
        'csi' : CriticalSuccessIndex,
        'CSI' : CriticalSuccessIndex,
    }
)