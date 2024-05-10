from enum import StrEnum, auto

class ProcessingMode(StrEnum):
    DeleteRow = auto()
    FillWithConstant = 'constant'
    FillWithTypeDefault = 'typedefault'
    FillWithAggregateFunction = 'aggregatefunction'