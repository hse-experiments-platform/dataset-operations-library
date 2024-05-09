import pandas as pd
import numpy as np
from typing import Dict
from sklearn.compose import ColumnTransformer
from .schema import *
from ..processing_mode import *
from ..aggregate_function import *
from .transform_empties import *

def apply_transformations(transformers, dataset):
    ct = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False).set_output(transform="pandas")

    ct.fit(dataset)
    transformed_dataset = ct.transform(dataset)

    return transformed_dataset

def map_column_type(column_type: str):
    if column_type == 'int':
        return 'int64'
    elif column_type == 'categorial':
        return 'string'
    elif column_type == 'float':
        return 'float64'
    else:
        raise ValueError('Unsupported column type')


def preprocess(dataset: pd.DataFrame, settings: Dict[str, DatasetPreprocessingSettings]):
    if set(dataset.columns) != set(settings.keys()):
        raise ValueError('Settings must be provided for all columns')

    dataset.replace(['', None, pd.NA], np.nan, inplace=True)

    columns = list(dataset)
    transformers = []

    for column in columns:
        column_type = settings[column].column_type
        empties_settings = settings[column].empties_settings
        
        transformer = get_empties_transformer(column, column_type, dataset, empties_settings)
        if transformer is not None:
            transformers.append(transformer)
   
    filled_dataset = apply_transformations(transformers, dataset)

    for column in columns:
        column_type = settings[column].column_type
        internal_column_type = map_column_type(column_type)
        try:
            filled_dataset[column] = filled_dataset[column].astype(internal_column_type)
        except ValueError:
            raise ValueError(f'Cannot convert column {column} to type {column_type}')

    return filled_dataset
