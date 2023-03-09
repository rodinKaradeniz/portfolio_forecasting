from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_summary(ticker: str, debug=False):
    if(ticker != ''):
        if debug:
            print("Scraping data...")

        driver = webdriver.Chrome()
        driver.get(f"https://finance.yahoo.com/quote/{ticker}?p={ticker}")
        assert "Yahoo" in driver.title

        update_time = time.strftime("%H:%M:%S", time.localtime())

        # to let the webpage load completely
        time.sleep(10)
        
        # Open up the page and extract summary information
        soup = BeautifulSoup(driver.page_source, 'lxml')
        tables = soup.find_all('table')
        df = pd.read_html(str(tables))[0]
    
        driver.quit()

        stock_summary = dict()
        for title, val in df:
            stock_summary[title] = val
        stock_summary['As of'] = update_time

        if debug:
            print(f"Scraping complete.")

        return stock_summary

    else:
        raise Exception("Invalid Ticker")


def scrape_history(ticker: str, debug=False):
    if(ticker != ''):
        if debug:
            print("Scraping data...")

        driver = webdriver.Chrome()
        driver.get(f"https://finance.yahoo.com/quote/{ticker}/history?p={ticker}")
        assert "Yahoo" in driver.title

        # to let the webpage load completely
        time.sleep(10)

        soup = BeautifulSoup(driver.page_source, 'lxml')
        tables = soup.find_all('table')
    
        dfs = pd.read_html(str(tables))
        dfs[0].to_csv(f'./history_data_raw/{ticker}.csv', index=False)
    
        driver.quit()

        if debug:
            print(f"Scraping complete. Data is saved in raw_data file as {ticker}.csv")

    else:
        raise Exception("Invalid Ticker")
    

if __name__ == "__main__":
    ticker = "AMZN"
    scrape_summary(ticker, debug=True)