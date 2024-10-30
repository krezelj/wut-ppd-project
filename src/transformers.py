from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)
    
class TimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def get_feature_names_out(self, feature_names_out):
        return [
            'time__day_of_year_sin', 
            'time__day_of_year_cos',
            'time__day_of_week',
            'time__time_of_day_sin',
            'time__time_of_day_cos',
            ]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X_copy = X.copy()

        def get_sin_cos(values, period):
            scaled = 2 * np.pi * values / period
            return np.sin(scaled), np.cos(scaled)

        dt = pd.to_datetime(X.flatten())
        day_of_year = dt.day_of_year
        day_of_week = dt.day_of_week
        time_of_day = dt.hour * 60 + dt.minute

        day_of_year_sin, day_of_year_cos = get_sin_cos(day_of_year, 366)
        time_of_day_sin, time_of_day_cos = get_sin_cos(time_of_day, 24*60)

        return pd.DataFrame(data={
            'day_of_year_sin': day_of_year_sin,
            'day_of_year_cos': day_of_year_cos,
            'day_of_week': day_of_week,
            'time_of_day_sin': time_of_day_sin,
            'time_of_day_cos': time_of_day_cos,
        })
    
class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X is a 2D array (samples, 1) of column `text`
        # get rid of @airline (pass possible names to init?)
        # enoce
        return X