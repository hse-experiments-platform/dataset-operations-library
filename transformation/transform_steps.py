import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from .schema import *
from .OutliersTransformer import *

def get_transform_name(transformer_name: str, column_name: str):
    return f'{transformer_name}_{column_name}'


def create_transformer(transform_name: str, transformer: any, column_name: str):
    return (get_transform_name(transform_name, column_name), transformer, [column_name])


def encode_enum_column(column_name, encoding_technique):
    if encoding_technique == EncodingTechnique.OneHotEncoding:
        return create_transformer('one_hot_encoder', OneHotEncoder(sparse_output=False), column_name)

    elif encoding_technique == EncodingTechnique.LabelEncoding:
        return create_transformer('ordinal_encoder', OrdinalEncoder(), column_name)
    
    else:
        raise ValueError('Unsupported encoding technique')


def scale_column(column_name, scaling_technique):
    if scaling_technique == ScalingTechnique.Standardization:
       return create_transformer('standardization', StandardScaler(), column_name)

    elif scaling_technique == ScalingTechnique.Normalization:
        return create_transformer('min_max_scaler', MinMaxScaler(), column_name)

    else:
        raise ValueError('Unsupported scaling technique')


def handle_outliers(column_name, outliers_strategy):
    detecting_mode = outliers_strategy.detecting_mode
    processing_mode = outliers_strategy.processing_mode

    transformer = OutliersTransformer(
        detecting_mode=detecting_mode,
        processing_mode=processing_mode,
        min_threshold=outliers_strategy.outlier_min,
        max_threshold=outliers_strategy.outlier_max,
        constant_value=outliers_strategy.constant_value,
        aggregate_function=outliers_strategy.aggregate_function
    )
    return create_transformer('outliers_transformer', transformer, column_name)
