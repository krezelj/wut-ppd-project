#!/bin/bash
curl -L -o ./data/archive.zip https://www.kaggle.com/api/v1/datasets/download/crowdflower/twitter-airline-sentiment
unzip ./data/archive.zip -d ./data
rm ./data/archive.zip