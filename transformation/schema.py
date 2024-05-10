from typing import Optional, Dict
from enum import StrEnum
from ..processing_mode import *
from ..aggregate_function import *

class OutliersDetectingMode(StrEnum):
    MinMax = auto()
    IQR = auto()

class EncodingTechnique(StrEnum):
    OneHotEncoding = 'OneHotEncoding'
    LabelEncoding = 'LabelEncoding'

class ScalingTechnique(StrEnum):
    Normalization = 'Normalization'
    Standardization = 'Standardization'


class OutliersStrategy:
    def __init__(self, detecting_mode: OutliersDetectingMode, processing_mode: ProcessingMode,
        outlier_min: Optional[float] = None, outlier_max: Optional[float] = None,
        constant_value: Optional[any] = None, aggregate_function: Optional[AggregateFunction] = None
    ):

        self.detecting_mode = detecting_mode
        self.outlier_min = outlier_min
        self.outlier_max = outlier_max
        self.processing_mode = processing_mode
        self.constant_value = constant_value
        self.aggregate_function = aggregate_function


class ColumnTransformSettings:
    def __init__(self,
        scaling_technique: Optional[ScalingTechnique] = None,
        encoding_technique: Optional[EncodingTechnique] = None,
        outliers_strategy: OutliersStrategy = None
    ):
        self.scaling_technique = scaling_technique
        self.encoding_technique = encoding_technique
        self.outliers_strategy = outliers_strategy