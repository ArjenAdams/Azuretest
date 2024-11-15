from enum import Enum


class CorrMatrixMethod(Enum):
    pearson = 1
    kendall = 2
    spearman = 3


class ImportanceType(Enum):
    gain = 1
    weight = 2
    cover = 3


class OtherVisualisation(Enum):
    TABLEONE = 1
    CORR_MATRIX = 2
    CONFUSION_MATRIX = 3
    FEATURE_IMPORTANCE = 4
    TREE_VISUALISATION = 5
    DATA_DRIFT = 6
    OUTLIER_DETECTION = 7
