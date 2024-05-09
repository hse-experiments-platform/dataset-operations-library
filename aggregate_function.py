from enum import Enum

class AggregateFunction(Enum):
    Max = 'Max'
    Min = 'Min'
    Average = 'Average'
    MostFrequent = 'MostFrequent'
    Median = 'Median'