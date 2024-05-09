from enum import Enum

class ProcessingMode(Enum):
    DeleteRow = 'DeleteRow'
    FillWithConstant = 'FillWithConstant'
    FillWithTypeDefault = 'FillWithTypeDefault'
    FillWithAggregateFunction = 'FillWithAggregateFunction'