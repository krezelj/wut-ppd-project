# Twitter US Airline Sentiment Analysis
Data processing course project for WUT 2024/2025 winter semester.
This repository compares classical Data Science methods with Neural Networks and pre-trained LLMs.

## Requirements
`data` directory should include the provided `database.sqlite` and `Tweets.csv` files. If it doesn't please use `download_dataset.sh` script. For that call:

```.sh
bash download_dataset.sh
```

Or download the dataset from [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).

## Run
To recreate the results please use the `data_processing_run_all.ipynb`. If there are issues with packages please run `pip install -r requirements.txt`.
