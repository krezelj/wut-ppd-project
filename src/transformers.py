import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

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
        self.vectorizer = TfidfVectorizer(max_features=200)

    def get_feature_names_out(self, feature_names_out):
        return [
            f'vectorizer__{name}' for name in self.vectorizer.get_feature_names_out()
        ]

    def fit(self, X, y=None):
        X = np.array(list(map(self._extract_tweet, X)))
        corpus = np.array(list(map(self._preprocess_tweet, X)))
        self.vectorizer.fit(corpus)
        return self
    
    def transform(self, X):
        corpus = np.array(list(map(self._extract_tweet, X)))
        return self.vectorizer.transform(corpus)
    
    def _extract_tweet(self, tweet):
        return tweet[0]
    
    def _preprocess_tweet(self, tweet):
        tweet = tweet.lower()

        # Remove URLs
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove mentions and hashtags
        tweet = re.sub(r"@\w+|#\w+", '', tweet)
        # Remove punctuation
        tweet = re.sub(r"[^\w\s]", '', tweet)
        # Remove numbers
        tweet = re.sub(r"[0-9]+", '', tweet)

        tokens = word_tokenize(tweet)
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)