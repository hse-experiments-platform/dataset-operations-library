import unittest
import numpy as np
import pandas as pd
from .transform import *
from .schema import *

class TransformTests(unittest.TestCase):

    def assertSeriesEqual(self, actual_series, expected_array):
        array = np.array(actual_series.values)
        self.assertTrue(np.allclose(array, expected_array, rtol=0.1))


    def test_ObjectInDatasetTypes_Error(self):
        dataset = pd.DataFrame({
            'age': [10, 20, 30, 40],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston']
        })

        with self.assertRaises(ValueError):
            transform(dataset, None)

    def test_scalingAndEncodingApplied(self):
        dataset = pd.DataFrame({
                'age': pd.Series([10, 20, 30, 40], dtype='int64'),
                'city': pd.Series(['New York', 'Los Angeles', 'Chicago', 'Houston'], dtype='string')
            }
        )
        settings = {
            'age': ColumnTransformSettings(
                scaling_technique=ScalingTechnique.Normalization,
            ),
            'city': ColumnTransformSettings(
                encoding_technique=EncodingTechnique.OneHotEncoding,
            )
        }

        transformed_dataset = transform(dataset, settings)

        self.assertSeriesEqual(transformed_dataset['age'], np.array([0, 0.33, 0.67, 1]))

        self.assertEqual(5, len(transformed_dataset.columns))
        self.assertSeriesEqual(transformed_dataset['city_New York'], np.array([1, 0, 0, 0]))
        self.assertSeriesEqual(transformed_dataset['city_Los Angeles'], np.array([0, 1, 0, 0]))
        self.assertSeriesEqual(transformed_dataset['city_Chicago'], np.array([0, 0, 1, 0]))
        self.assertSeriesEqual(transformed_dataset['city_Houston'], np.array([0, 0, 0, 1]))

    def test_outliersHandlingApplied(self):
        dataset = pd.DataFrame({
                'age': pd.Series([10, 20, 30, 1000], dtype='int64')
            }
        )
        settings = {
            'age': ColumnTransformSettings(
                scaling_technique=ScalingTechnique.Normalization,
                outliers_strategy=OutliersStrategy(
                    detecting_mode=OutliersDetectingMode.MinMax,
                    outlier_min=0,
                    outlier_max=40,
                    processing_mode=ProcessingMode.FillWithConstant,
                    constant_value=40
                )
            )
        }

        transformed_dataset = transform(dataset, settings)

        self.assertSeriesEqual(transformed_dataset['age'], np.array([0, 0.33, 0.67, 1]))


if __name__ == '__main__':
    unittest.main()