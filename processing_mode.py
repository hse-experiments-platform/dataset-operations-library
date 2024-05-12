from enum import StrEnum, auto

class ProcessingMode(StrEnum):
    DeleteRow = 'DeleteRow'
    Constant = 'Constant'
    TypeDefault = 'TypeDefault'
    AggregateFunction = 'AggregateFunction'