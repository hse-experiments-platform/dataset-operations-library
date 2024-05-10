from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .schema import *
from .transform_steps import *


def apply_transformations(scale_encode_transformers, outliers_transformers, dataset):
    set_config(transform_output = "pandas")
    ct1 = ColumnTransformer(outliers_transformers, remainder='passthrough', verbose_feature_names_out=False)
    ct2 = ColumnTransformer(scale_encode_transformers, remainder='passthrough', verbose_feature_names_out=False)

    estimators = [
        ('outliers', ct1),
        ('scale_encode', ct2)
    ]
    pipeline = Pipeline(steps=estimators)

    transformed_dataset = pipeline.fit_transform(dataset)

    return transformed_dataset


def transform(dataset, settings: Dict[str, ColumnTransformSettings]):
    dataset_types = dataset.dtypes
    if 'object' in dataset_types.values:
        raise ValueError("Dataset contains object dtype columns.")
    
    scale_encode_transformers = []
    outliers_transformers = []

    for column_name in settings:
        column_type = dataset_types[column_name]
        column_settings = settings[column_name]
        
        scale_enum_transformer = None
        if column_type == 'string':
            scale_enum_transformer = encode_enum_column(column_name, column_settings.encoding_technique)
        else:
            scale_enum_transformer = scale_column(column_name, column_settings.scaling_technique)

            if column_settings.outliers_strategy is not None:
                outliers_transformer = handle_outliers(column_name, column_settings.outliers_strategy)
                outliers_transformers.append(outliers_transformer)
        scale_encode_transformers.append(scale_enum_transformer)

    transformed_dataset = apply_transformations(scale_encode_transformers, outliers_transformers, dataset)
    return transformed_dataset

