import unittest
import pandas as pd
import numpy as np
from .schema import *
from .preprocess import *

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
        'a': DatasetPreprocessingSettings('int', EmptiesStrategy(ProcessingMode.FillWithAggregateFunction, aggregate_function=AggregateFunction.Max)),
        'b': DatasetPreprocessingSettings('categorial', EmptiesStrategy(ProcessingMode.DeleteRow)),
        'c': DatasetPreprocessingSettings('float', EmptiesStrategy(ProcessingMode.FillWithAggregateFunction, aggregate_function=AggregateFunction.Min)),
    }
    return settings

class PreprocessTests(unittest.TestCase):
    
    def test_notAllColumnsArePresentInSettings_Error(self):
        dataset = get_dataset()

        settings = {
            'a': DatasetPreprocessingSettings('int', EmptiesStrategy(ProcessingMode.FillWithAggregateFunction, aggregate_function=AggregateFunction.Max)),
            'b': DatasetPreprocessingSettings('categorial', EmptiesStrategy(ProcessingMode.DeleteRow)),
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
            'a': DatasetPreprocessingSettings('unsupported', EmptiesStrategy(ProcessingMode.FillWithAggregateFunction, aggregate_function=AggregateFunction.Max)),
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
            'a': DatasetPreprocessingSettings('int', EmptiesStrategy(ProcessingMode.FillWithTypeDefault)),
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


if __name__ == '__main__':
    unittest.main()
