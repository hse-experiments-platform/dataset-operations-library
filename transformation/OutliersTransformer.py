from sklearn.base import BaseEstimator, TransformerMixin
from ..processing_mode import *
from ..aggregate_function import *
from .schema import *

class OutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, detecting_mode=OutliersDetectingMode.IQR, processing_mode=ProcessingMode.DeleteRow,
        min_threshold=None, max_threshold=None,
        constant_value=None, aggregate_function=None
    ):
        if detecting_mode == OutliersDetectingMode.MinMax and (min_threshold is None or max_threshold is None):
            raise ValueError('min_threshold and max_threshold must be specified for minmax method')
        if processing_mode == ProcessingMode.Constant and constant_value is None:
            raise ValueError('constant_value must be specified for constant processing mode')
        if processing_mode == ProcessingMode.AggregateFunction and aggregate_function is None:
            raise ValueError('aggregate_functiong must be specified for aggregatefunction processing mode')
 
        self.detecting_mode = detecting_mode
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.processing_mode = processing_mode
        self.constant_value = constant_value
        self.aggregate_function = str(aggregate_function).lower() if aggregate_function else None

        self.threshold = 1.5
        self.outliers = None


    def fit(self, X, y=None):
        return self


    def _detect_outliers(self, X):
        if self.detecting_mode == OutliersDetectingMode.IQR:
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            self.outliers = (X < (Q1 - self.threshold * IQR)) | (X > (Q3 + self.threshold * IQR))
        elif self.detecting_mode == OutliersDetectingMode.MinMax:
            self.outliers = (X < self.min_threshold) | (X > self.max_threshold)
        else:
            raise ValueError('Unsupported detecting mode')

        return self


    def _process_outliers(self, X):
        if self.outliers is None:
            return X

        if self.processing_mode == ProcessingMode.DeleteRow:
            return X[~self.outliers]
        elif self.processing_mode == ProcessingMode.Constant:
            X[self.outliers] = self.constant_value
            return X
        elif self.processing_mode == ProcessingMode.TypeDefault:
            X[self.outliers] = 0
            return X
        elif self.processing_mode == ProcessingMode.AggregateFunction:
            X_exclude_outliers = X[~self.outliers]
            aggregation_value = X_exclude_outliers.agg(self.aggregate_function)
            X[self.outliers] = aggregation_value
            return X
        else:
            raise ValueError('Unsupported processing mode')


    def transform(self, X):
        self._detect_outliers(X)
        result = self._process_outliers(X)
        return result
       