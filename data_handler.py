from data_processing.data_scraper import *
from data_processing.data_processor import *
import os
import pandas as pd

def get_stock_summary(ticker):
    # No need to introduce easy saving or caching here - this is momentary data
    return scrape_summary(ticker)

def get_stock_history(ticker, save_to_db=False):
    # If history has already been pulled, get data
    if os.path.isfile(f"./data_processing/history_data/{ticker}.csv"):
        df = pd.read_csv(f"./data_processing/history_data/{ticker}.csv", index=False)

    else:
        # Scrape data from yahoo.com
        df = scrape_history(ticker)

        # Clean/Transform data
        df = refactor_dataframe(df)

        # Save as csv for future access
        df.to_csv(f'./data_processing/history_data/{ticker}.csv', index=False)

    if save_to_db:
        # Save to the database
        db_name = "stock_data.db"
        upload_to_sql(db_name, df)

    return df
