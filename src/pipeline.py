from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from .transformers import *

columns_to_drop = ['retweet_count', 'airline_sentiment_gold', 'negativereason_gold', 
                   'tweet_coord', 'name', 'user_timezone', 'negativereason', 'negativereason_confidence']
columns_to_fill_zero = []
columns_to_fill_unknown = ['tweet_location']
columns_to_ohe = ['airline', 'tweet_location']

column_order_after_transform = \
    columns_to_fill_zero + columns_to_fill_unknown + ['airline', 'text', 'tweet_created']
column_idx = lambda c: column_order_after_transform.index(c)

preprocessor = Pipeline(steps=[
    ('drop', DropColumnTransformer(columns_to_drop)),
    ('fill_missing',
        ColumnTransformer(
            transformers=[
                ('fill_zero', SimpleImputer(strategy='constant', fill_value=0), columns_to_fill_zero),
                ('fill_other', SimpleImputer(strategy='constant', fill_value='Unknown'), columns_to_fill_unknown),
            ],
            remainder='passthrough')),
    ('encode', ColumnTransformer(transformers=[
        ('ohe', OneHotEncoder(
            handle_unknown='infrequent_if_exist',
            max_categories=10,
            sparse_output=False),
            list(map(column_idx, columns_to_ohe))),
        ('time', TimeTransformer(), list(map(column_idx, ['tweet_created']))),
        ('text', TextTransformer(), list(map(column_idx, ['text'])))
    ],
    remainder='passthrough'))
])