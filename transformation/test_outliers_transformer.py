import unittest
import pandas as pd
from .OutliersTransformer import *
from ..processing_mode import *
from ..aggregate_function import *
from .schema import *

class OutliersTransformerTests(unittest.TestCase):
    
    def test_DetectingModeMinMax_WithoutThresholds_Error(self):
        with self.assertRaises(ValueError):
            transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.MinMax)

    def test_ProcessingModeConstant_WithoutConstantValue_Error(self):
        with self.assertRaises(ValueError):
            transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.Constant)

    def test_ProcessingModeAggregateFunction_WithoutAggregateFunction_Error(self):
        with self.assertRaises(ValueError):
            transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.AggregateFunction)

    def test_DetectingModeIQR_OutliersDetectedAndRemoved(self):
        transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.DeleteRow)
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = transformer.fit_transform(X)
        self.assertEqual(len(result), 8)

    def test_DetectingModeMinMax_OutliersDetectedAndRemoved(self):
        transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.MinMax, processing_mode=ProcessingMode.DeleteRow, min_threshold=3, max_threshold=7)
        X = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = transformer.fit_transform(X)
        self.assertEqual(len(result), 5)

    def test_ProcessingModeConstant_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.Constant, constant_value=3)
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = transformer.fit_transform(X)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 3)
        self.assertEqual(result[9], 3)

    def test_ProcessingModeTypeDefault_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.TypeDefault)
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = transformer.fit_transform(X)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[9], 0)

    def test_ProcessingModeAggregateFunction_WithMean_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Mean)
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype='float64')

        result = transformer.fit_transform(X)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 5.5)
        self.assertEqual(result[9], 5.5)

    def test_ProcessingModeAggregateFunction_WithMedian_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Median)
        X = pd.Series([-100, 2, 3, 4, 5, 5, 6, 7, 8, 9, 100], dtype='float64')

        result = transformer.fit_transform(X)

        self.assertEqual(len(result), 11)
        self.assertEqual(result[0], 5)
        self.assertEqual(result[10], 5)

    def test_ProcessingModeAggregateFunction_WithMax_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Max)
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype='float64')

        result = transformer.fit_transform(X)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 9)
        self.assertEqual(result[9], 9)

    def test_ProcessingModeAggregateFunction_WithMin_OutliersReplaced(self):
        transformer = OutliersTransformer(detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Min)
        X = pd.Series([-100, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype='float64')

        result = transformer.fit_transform(X)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[9], 2)


if __name__ == '__main__':
    unittest.main()