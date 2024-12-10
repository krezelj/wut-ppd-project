from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from .transformers import *

# Define columns for different preprocessing steps
columns_to_drop = ['retweet_count', 'airline_sentiment_gold', 'negativereason_gold', 'tweet_coord', 'name', 'user_timezone']
columns_to_fill_zero = ['negativereason_confidence']
columns_to_fill_unknown = ['negativereason', 'tweet_location']
columns_to_ohe = ['negativereason', 'airline', 'tweet_location']

# Define the order of columns after transformation
column_order_after_transform = \
    columns_to_fill_zero + columns_to_fill_unknown + ['airline', 'text', 'tweet_created']
column_idx = lambda c: column_order_after_transform.index(c)

# Create a preprocessing pipeline
preprocessor = Pipeline(steps=[
    # Step 1: Drop unnecessary columns
    ('drop', DropColumnTransformer(columns_to_drop)),
    # Step 2: Fill missing values
    ('fill_missing',
        ColumnTransformer(
            transformers=[
                ('fill_zero', SimpleImputer(strategy='constant', fill_value=0), columns_to_fill_zero),
                ('fill_other', SimpleImputer(strategy='constant', fill_value='Unknown'), columns_to_fill_unknown),
            ],
            remainder='passthrough')),
    # Step 3: Encode categorical variables and transform other features
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