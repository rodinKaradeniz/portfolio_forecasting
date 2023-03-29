import os
from data_processing.data_scraper import *
from data_processing.data_processor import *
from sentiment_analysis.train import predict

def get_sentiment_prediction(ticker):
    # TODO: Implement

    # Get data from DB
    # TODO: Introduce DB into this system

    # Get relevant articles / tweets
    # TODO: Twitter API
    # TODO: Scrape relevant news

    # Predict
    # TODO: Predict tweets using predict()
    pass

def get_price_prediction(ticker):
    # TODO: Implement

    # Get data from DB
    load_from_sql(ticker, debug=False)

    # Preprocess data

    # Train model if not trained already

    # Predict

    pass