from data_processing.data_scraper import *
from data_processing.data_processor import *
import os
import pandas as pd

def get_stock_summary(ticker):
    # No need to introduce saving/caching here - this is momentary data
    return scrape_summary(ticker)

def get_stock_history(ticker, save_to_db=False, debug=False):
    # If history has already been pulled, get data
    if os.path.isfile(f"./data/history_data/{ticker}.csv"):
        df = pd.read_csv(f"./data/history_data/{ticker}.csv", index=False)

    # Else, extract data and transform
    else:
        # Scrape data from yahoo.com
        scrape_history(ticker)

        # Retrieve raw data
        df = pd.read_csv('./data/history_data_raw/{ticker}.csv', index=False)

        # Clean/Transform data
        df = refactor_dataframe(df)

        # Save as csv for future access
        df.to_csv(f'./data/history_data/{ticker}.csv', index=False)

    if save_to_db:
        # Load to the database
        db_name = "{ticker}_stock_history.db"
        upload_to_sql(db_name, df)

    return df
