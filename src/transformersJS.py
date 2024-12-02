from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import string
import re
import emoji
import nltk
import re
import emoji
import nltk

nltk.download("omw-1.4")
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("wordnet")
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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
            "time__day_of_year_sin",
            "time__day_of_year_cos",
            "time__day_of_week",
            "time__time_of_day_sin",
            "time__time_of_day_cos",
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
        time_of_day_sin, time_of_day_cos = get_sin_cos(time_of_day, 24 * 60)

        return pd.DataFrame(
            data={
                "day_of_year_sin": day_of_year_sin,
                "day_of_year_cos": day_of_year_cos,
                "day_of_week": day_of_week,
                "time_of_day_sin": time_of_day_sin,
                "time_of_day_cos": time_of_day_cos,
            }
        )


class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X.flatten()
        processed_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = ""
            # we want to delte all urls, mentions and emojis
            # text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE) # does not work (???)
            text = re.sub(r"@\w+", "", text)
            # or text = re.sub(r"@\w+", "@airline", text) to include that this is an airline
            text = emoji.demojize(text)
            text = text.lower()
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            processed_text = " ".join(tokens)
            processed_texts.append(processed_text)
        self.processed_texts_ = processed_texts

        return processed_texts
