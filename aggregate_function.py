from enum import StrEnum

class AggregateFunction(StrEnum):
    Max = 'Max'
    Min = 'Min'
    Mean = 'Mean'
    MostFrequent = 'MostFrequent'
    Median = 'Median'