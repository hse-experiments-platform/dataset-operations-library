import unittest
import pandas as pd
import numpy as np
from .schema import *
from .preprocess import *
from ..accepted_types import *

def get_dataset():
    dataset = pd.DataFrame(
        {
            'a': pd.Series([1, np.nan, 3]),
            'b': pd.Series(['one', 'two', np.nan]),
            'c': pd.Series([5.2, np.nan, 4.19]),
        }
    )
    return dataset

def get_settings():
    settings = {
        'a': DatasetPreprocessingSettings(AcceptedType.Int,
                EmptiesStrategy(ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Max)),
        'b': DatasetPreprocessingSettings(AcceptedType.Categorial,
                EmptiesStrategy(ProcessingMode.DeleteRow)),
        'c': DatasetPreprocessingSettings(AcceptedType.Float,
                EmptiesStrategy(ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Min)),
    }
    return settings

class PreprocessTests(unittest.TestCase):
    
    def test_notAllColumnsArePresentInSettings_Error(self):
        dataset = get_dataset()

        settings = {
            'a': DatasetPreprocessingSettings(AcceptedType.Int,
                    EmptiesStrategy(ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Max)),
            'b': DatasetPreprocessingSettings(AcceptedType.Categorial,
                    EmptiesStrategy(ProcessingMode.DeleteRow)),
        }

        with self.assertRaises(ValueError):
            preprocess(dataset, settings)

    def test_emptyValuesAreAbsent(self):
        dataset = get_dataset()
        settings = get_settings()

        result = preprocess(dataset, settings)

        self.assertFalse(result.isnull().values.any())

    def test_correctColumnTypes(self):
        dataset = get_dataset()
        settings = get_settings()

        result = preprocess(dataset, settings)

        self.assertEqual(result['a'].dtype, 'int64')
        self.assertEqual(result['b'].dtype, 'string')
        self.assertEqual(result['c'].dtype, 'float64')


    def test_unsupportedColumnType_Error(self):
        dataset = pd.DataFrame(
            {
                'a': pd.Series([1, np.nan, 3]),
            }
        )
        settings = {
            'a': DatasetPreprocessingSettings('unsupported', EmptiesStrategy(ProcessingMode.AggregateFunction, aggregate_function=AggregateFunction.Max)),
        }

        with self.assertRaises(ValueError):
            preprocess(dataset, settings)

    
    def test_incorrectDataType_Error(self):
        dataset = pd.DataFrame(
            {
                'a': pd.Series(['a', np.nan, 'b']),
            }
        )
        settings = {
            'a': DatasetPreprocessingSettings(AcceptedType.Int, EmptiesStrategy(ProcessingMode.TypeDefault)),
        }

        with self.assertRaises(ValueError):
            preprocess(dataset, settings)

    
    def test_noneValuesAreReplaced(self):
        dataset = pd.DataFrame(
            {
                'a': pd.Series([1, None, 3]),
                'b': pd.Series(['one', 'two', None]),
                'c': pd.Series([5.2, None, 4.19]),
            }
        )
        settings = get_settings()

        result = preprocess(dataset, settings)

        self.assertFalse(result.isnull().values.any())

    def test_columnIsDroped(self):
        dataset = get_dataset()
        settings = {
            'a': DatasetPreprocessingSettings(AcceptedType.Int, EmptiesStrategy(ProcessingMode.TypeDefault)),
            'b': DatasetPreprocessingSettings(AcceptedType.Dropped, None),
            'c': DatasetPreprocessingSettings(AcceptedType.Float, EmptiesStrategy(ProcessingMode.TypeDefault)),
        }

        result = preprocess(dataset, settings)

        self.assertTrue('a' in result.columns)
        self.assertFalse('b' in result.columns)
        self.assertTrue('c' in result.columns)

    def test_initialOrderSaved(self):
        dataset = get_dataset()
        settings = get_settings()

        result = preprocess(dataset, settings)

        self.assertEqual(list(dataset.columns), list(result.columns))


if __name__ == '__main__':
    unittest.main()
