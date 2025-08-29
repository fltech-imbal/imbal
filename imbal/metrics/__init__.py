from .TruePositiveRate import TruePositiveRate
from .FalsePositiveRate import FalsePositiveRate
from .TrueNegativeRate import TrueNegativeRate
from .ExpectedTruePositive import ExpectedTruePositive
from .ExpectedTrueNegative import ExpectedTrueNegative
from .ExpectedCorrect import ExpectedCorrect
from .TrueSkillStatistic import TrueSkillStatistic
from .HeikdeSkillScore import HeikdeSkillScore
from .GilbertSkillScore import GilbertSkillScore
from .CriticalSuccessIndex import CriticalSuccessIndex
from .LimitedAUC import LimitedAUC

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
    LimitedAUC
}

IMBAL_OBJECTS_DICT = {cls.__name__: cls for cls in IMBAL_OBJECTS}
IMBAL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in IMBAL_OBJECTS}
)
IMBAL_OBJECTS_DICT.update(
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
        'hss' : HeikdeSkillScore,
        'HSS' : HeikdeSkillScore,
        'gss' : GilbertSkillScore,
        'GSS' : GilbertSkillScore,
        'gs' : GilbertSkillScore,
        'GS' : GilbertSkillScore,
        'csi' : CriticalSuccessIndex,
        'CSI' : CriticalSuccessIndex,
    }
)
ALL_OBJECTS_DICT.update(IMBAL_OBJECTS_DICT)