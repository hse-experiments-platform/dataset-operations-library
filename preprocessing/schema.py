from typing import Optional, Dict
from ..processing_mode import *
from ..aggregate_function import *

class EmptiesStrategy:
    def __init__(self, technique: ProcessingMode, constant_value: Optional[any] = None, aggregate_function: Optional[AggregateFunction] = None):
        if technique == ProcessingMode.FillWithConstant and constant_value is None:
            raise ValueError('Constant value must be provided for FillWithConstant technique')
        if technique == ProcessingMode.FillWithAggregateFunction and aggregate_function is None:
            raise ValueError('Aggregate function must be provided for FillWithAggregateFunction technique')
        
        self.technique = technique
        self.constant_value = constant_value
        self.aggregate_function = aggregate_function

    def is_mode_available_for_enum(self):
        if self.technique == ProcessingMode.FillWithTypeDefault:
            return False
        
        if self.technique == ProcessingMode.FillWithAggregateFunction and self.aggregate_function != AggregateFunction.MostFrequent:
            return False
        
        return True

    def is_mode_available_for_all_nulls(self):
        return self.technique in [ProcessingMode.FillWithConstant, ProcessingMode.FillWithTypeDefault]


class DatasetPreprocessingSettings():
    def __init__(self, column_type: str, empties_settings: EmptiesStrategy):
        self.column_type = column_type
        self.empties_settings = empties_settings