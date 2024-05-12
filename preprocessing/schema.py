from typing import Optional, Dict
from ..processing_mode import *
from ..aggregate_function import *
from ..accepted_types import *

class EmptiesStrategy:
    def __init__(self, technique: ProcessingMode, constant_value: Optional[any] = None, aggregate_function: Optional[AggregateFunction] = None):
        if technique == ProcessingMode.Constant and constant_value is None:
            raise ValueError('Constant value must be provided for Constant technique')
        if technique == ProcessingMode.AggregateFunction and aggregate_function is None:
            raise ValueError('Aggregate function must be provided for AggregateFunction technique')
        
        self.technique = technique
        self.constant_value = constant_value
        self.aggregate_function = aggregate_function

    def is_mode_available_for_enum(self):
        if self.technique == ProcessingMode.TypeDefault:
            return False
        
        if self.technique == ProcessingMode.AggregateFunction and self.aggregate_function != AggregateFunction.MostFrequent:
            return False
        
        return True

    def is_mode_available_for_all_nulls(self):
        return self.technique in [ProcessingMode.Constant, ProcessingMode.TypeDefault]


class DatasetPreprocessingSettings():
    def __init__(self, column_type: AcceptedType, empties_settings: EmptiesStrategy):
        self.column_type = column_type
        self.empties_settings = empties_settings