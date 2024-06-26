import unittest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from ..processing_mode import *
from ..accepted_types import *
from .transform_empties import *
from .schema import *

def get_transform_result(dataset, transformer):
    assert transformer is not None
    return transformer[1].fit_transform(dataset)


class TestProcessEmptyValues(unittest.TestCase):
    def test_datasetContainsUnprocessableEmptyValues_Error(self):
        dataset = pd.DataFrame({'a': pd.Series([1, '', 3])})
        with self.assertRaises(ValueError):
            get_empties_transformer('a', AcceptedType.Int, dataset, ProcessingMode.DeleteRow)

    def test_allValuesAreNullAndModeIsNotSupported_Error(self):
        dataset = pd.DataFrame({'a': pd.Series([np.nan, np.nan])})
        empties_settings = EmptiesStrategy(ProcessingMode.DeleteRow)
        with self.assertRaises(ValueError):
            get_empties_transformer('a', AcceptedType.Int, dataset, empties_settings)

    def test_processEnumAndModeIsNotSupported_Error(self):
        dataset = pd.DataFrame({'a': pd.Series(['a', 'b', 'c'])})
        empties_settings = EmptiesStrategy(ProcessingMode.TypeDefault)
        with self.assertRaises(ValueError):
            get_empties_transformer('a', AcceptedType.Categorial, dataset, empties_settings)

    def test_deleteRow(self):
        dataset = pd.DataFrame({'a': pd.Series([1, np.nan, 3])})
        empties_settings = EmptiesStrategy(ProcessingMode.DeleteRow)
        
        transformer = get_empties_transformer('a', AcceptedType.Int, dataset, empties_settings)

        self.assertIsNone(transformer)
        self.assertEqual(2, len(dataset))

    def test_Constant(self):
        dataset = pd.DataFrame({'a': pd.Series([1, np.nan, 3])})
        empties_settings = EmptiesStrategy(ProcessingMode.Constant, 5)
        
        transformer = get_empties_transformer('a', AcceptedType.Int, dataset, empties_settings)
        result = get_transform_result(dataset, transformer)

        self.assertEqual(5, result[1][0])

    def test_TypeDefault(self):
        dataset = pd.DataFrame({'a': pd.Series([1, np.nan, 3])})
        empties_settings = EmptiesStrategy(ProcessingMode.TypeDefault)
        
        transformer = get_empties_transformer('a', AcceptedType.Int, dataset, empties_settings)
        result = get_transform_result(dataset, transformer)

        self.assertEqual(0, result[1][0])

    def test_fillWithMax(self):
        dataset = pd.DataFrame({'a': pd.Series([1, np.nan, 3])})
        empties_settings = EmptiesStrategy(ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Max)
        
        transformer = get_empties_transformer('a', AcceptedType.Int, dataset, empties_settings)
        result = get_transform_result(dataset, transformer)

        self.assertEqual(3, result[1][0])

    def test_fillWithMin(self):
        dataset = pd.DataFrame({'a': pd.Series([1, np.nan, 3])})
        empties_settings = EmptiesStrategy(ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Min)
        
        transformer = get_empties_transformer('a', AcceptedType.Int, dataset, empties_settings)
        result = get_transform_result(dataset, transformer)

        self.assertEqual(1, result[1][0])

    def test_fillWithMean(self):
        dataset = pd.DataFrame({'a': pd.Series([1, np.nan, 3])})
        empties_settings = EmptiesStrategy(ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Mean)
        
        transformer = get_empties_transformer('a', AcceptedType.Int, dataset, empties_settings)
        result = get_transform_result(dataset, transformer)

        self.assertEqual(2, result[1][0])

    def test_fillWithMostFrequent(self):
        dataset = pd.DataFrame({'a': pd.Series(['Moscow', 'Barcelona', np.nan, 'Moscow'])})
        empties_settings = EmptiesStrategy(ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.MostFrequent)
        
        transformer = get_empties_transformer('a', AcceptedType.Categorial, dataset, empties_settings)
        result = get_transform_result(dataset, transformer)

        self.assertEqual('Moscow', result[2][0])

    def test_fillWithMedian(self):
        dataset = pd.DataFrame({'a': pd.Series([1, np.nan, 3])})
        empties_settings = EmptiesStrategy(ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Median)
        
        transformer = get_empties_transformer('a', AcceptedType.Int, dataset, empties_settings)
        result = get_transform_result(dataset, transformer)

        self.assertEqual(2, result[1][0])

if __name__ == '__main__':
    unittest.main()
