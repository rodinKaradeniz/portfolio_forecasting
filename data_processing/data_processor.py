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

def upload_to_sql(db_name: str, df: pd.DataFrame, debug=False):
    # Connect to the database
    conn = sqlite3.connect(f"{db_name}.db")
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

def etl(ticker: str, db_name: str, debug=False):
    df = pd.read_csv(f'./history_data_raw/{ticker}.csv')
    df = refactor_dataframe(df)

    # TODO: Handle the directory adjustment for database creation/update.
    # if os.path.isfile(f"../{db_name}.db"):
    #     upload_to_sql(db_name, df, debug)
    # else:
    #     raise Exception("Database does not exist")
    return