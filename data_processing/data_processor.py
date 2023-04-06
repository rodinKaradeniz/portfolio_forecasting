from collections import deque
from tqdm.auto import tqdm

import os.path
import pandas as pd
import sqlite3


def refactor_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # Add ticker as a new column
    df['ticker'] = ticker

    # Drop last row
    df.drop(df.index[-1], inplace = True)

    # Remove duplicates
    df.drop_duplicates(inplace = True)

    # Remove na values
    df.dropna(inplace = True)

    # Relabel Columns
    relabel = {
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close*': 'close',
        'Adj Close**': 'adj_close',
        'Volume': 'volume'
        }
    
    for old, new in relabel:
        df.rename(columns = {old: new}, inplace = True)

    # Remove "Dividend" rows if exist
    rows_to_remove = deque()
    for index, row in df.iterrows():
        if 'Dividend' in row['open']:
            rows_to_remove.append(index)
    df = df.drop(rows_to_remove)

    # Convert types of columns
    df['date'] = pd.to_datetime(df['date'])
    df['open'] = df['open'].astype('float')
    df['high'] = df['high'].astype('float')
    df['low'] = df['low'].astype('float')
    df['close'] = df['close'].astype('float')
    df['adj_close'] = df['adj_close'].astype('float')
    df['volume'] = df['volume'].astype('int')

    return df


def upload_to_sql(ticker: str, df: pd.DataFrame, debug=False):
    # Check if database exists
    db_name = f"./data/{ticker}.db"
    if os.path.isfile(db_name):
        conn = sqlite3.connect(db_name)
    else:
        # TODO: initialize db_name

        raise Exception("Database does not exist")

    columns = [
        "ticker TEXT",
        "date FLOAT",
        "open FLOAT",
        "high FLOAT",
        "low FLOAT",
        "close FLOAT",
        "adj_close FLOAT",
        "volume INT"
    ]
    create_table_cmd = f"CREATE TABLE IF NOT EXISTS stock ({','.join(columns)})"
    conn.execute(create_table_cmd)

    if debug:
        print("Uploading into database...")

    df.to_sql(db_name, con=conn, schema='stock', index=False, if_exists='append')
    
    if debug:
        print("Upload completed.")

    conn.commit()
    conn.close()


def load_from_sql(ticker: str, debug=False):
    db_name = f"{ticker}_stock_history.db"

    if os.path.isfile(f"../{db_name}.db"):
        if debug:
            print("Loading data from database...")

        conn = sqlite3.connect(f"{db_name}.db")

        # TODO: Load Data
    else:
        raise Exception("Database does not exist")