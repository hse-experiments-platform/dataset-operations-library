import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from .schema import *
from ..aggregate_function import *
from ..processing_mode import *
from ..accepted_types import *

def get_transform_name(transformer_name: str, column_name: str):
    return f'{transformer_name}_{column_name}'


def create_transformer(strategy: str, column_name: str, fill_value: any = None):
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    return (get_transform_name(strategy, column_name), imputer, [column_name])


def get_empties_transformer(column_name: str, column_type: AcceptedTypes, dataset: pd.DataFrame, empties_strategy: EmptiesStrategy):
    if dataset[column_name].isin(['', None, pd.NA]).any():
        raise ValueError("Dataset contains not processable empty values.")
    if dataset[column_name].isnull().all() and not empties_strategy.is_mode_available_for_all_nulls():
        raise ValueError("Cannot replace empty values when all values are null.")

    if column_type == AcceptedTypes.Enum and not empties_strategy.is_mode_available_for_enum():
        raise ValueError("Cannot fill enum type with default value.")

    if empties_strategy.technique == ProcessingMode.DeleteRow:
        dataset.dropna(subset=[column_name], inplace=True)
        return None
    
    elif empties_strategy.technique == ProcessingMode.FillWithConstant:
        return create_transformer('constant', column_name, empties_strategy.constant_value)

    elif empties_strategy.technique == ProcessingMode.FillWithTypeDefault:
        str(column_type)
        column_default = eval(str(column_type))()
        return create_transformer('constant', column_name, column_default)

    elif empties_strategy.technique == ProcessingMode.FillWithAggregateFunction:
        if empties_strategy.aggregate_function == AggregateFunction.Max:
            max_value = dataset[column_name].max()
            return create_transformer('constant', column_name, max_value)
        
        elif empties_strategy.aggregate_function == AggregateFunction.Min:
            min_value = dataset[column_name].min()
            return create_transformer('constant', column_name, min_value)

        elif empties_strategy.aggregate_function == AggregateFunction.Average:
            return create_transformer('mean', column_name)

        elif empties_strategy.aggregate_function == AggregateFunction.MostFrequent:
            return create_transformer('most_frequent', column_name)

        elif empties_strategy.aggregate_function == AggregateFunction.Median:
            return create_transformer('median', column_name)
