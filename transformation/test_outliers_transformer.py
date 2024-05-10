import unittest
import pandas as pd
from .OutliersTransformer import *

class OutliersTransformerTests(unittest.TestCase):
    
    def test_DetectingModeMinMax_WithoutThresholds_Error(self):
        with self.assertRaises(ValueError):
            transformer = OutliersTransformer(detecting_mode='minmax')

    def test_ProcessingModeConstant_WithoutConstantValue_Error(self):
        with self.assertRaises(ValueError):
            transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='constant')

    def test_ProcessingModeAggregateFunction_WithoutAggregateFunction_Error(self):
        with self.assertRaises(ValueError):
            transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='aggregatefunction')

    def test_DetectingModeIQR_OutliersDetectedAndRemoved(self):
        transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='deleterow')
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = transformer.fit_transform(X)
        self.assertEqual(len(result), 8)

    def test_DetectingModeMinMax_OutliersDetectedAndRemoved(self):
        transformer = OutliersTransformer(detecting_mode='minmax', processing_mode='deleterow', min_threshold=3, max_threshold=7)
        X = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = transformer.fit_transform(X)
        self.assertEqual(len(result), 5)

    def test_ProcessingModeConstant_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='constant', constant_value=3)
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = transformer.fit_transform(X)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 3)
        self.assertEqual(result[9], 3)

    def test_ProcessingModeTypeDefault_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='typedefault')
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = transformer.fit_transform(X)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[9], 0)

    def test_ProcessingModeAggregateFunction_WithMean_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='aggregatefunction', aggregate_function='mean')
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype='float64')

        result = transformer.fit_transform(X)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 5.5)
        self.assertEqual(result[9], 5.5)

    def test_ProcessingModeAggregateFunction_WithMedian_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='aggregatefunction', aggregate_function='median')
        X = pd.Series([-100, 2, 3, 4, 5, 5, 6, 7, 8, 9, 100], dtype='float64')

        result = transformer.fit_transform(X)

        self.assertEqual(len(result), 11)
        self.assertEqual(result[0], 5)
        self.assertEqual(result[10], 5)

    def test_ProcessingModeAggregateFunction_WithSum_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='aggregatefunction', aggregate_function='sum')
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype='float64')

        result = transformer.fit_transform(X)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 44)
        self.assertEqual(result[9], 44)

    def test_ProcessingModeAggregateFunction_WithMax_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='aggregatefunction', aggregate_function='max')
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype='float64')

        result = transformer.fit_transform(X)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 9)
        self.assertEqual(result[9], 9)

    def test_ProcessingModeAggregateFunction_WithMin_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode='iqr', processing_mode='aggregatefunction', aggregate_function='min')
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype='float64')

        result = transformer.fit_transform(X)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[9], 2)


if __name__ == '__main__':
    unittest.main()