import yfinance as yf
import pandas as pd
import os
from os.path import join
from datetime import datetime, timezone
import time
import logging

def load_ticker_list(pth: str) -> list[str]:
    ticker_list = pd.read_csv(pth)['ticker_code'].to_list()
    return ticker_list

RawDF_Schema = {
    'Date': 'datetime64[ns, Europe/Berlin]',
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Close': 'float64',
    'Volume': 'int64',
    'Dividends': 'float64',
    'Stock Splits': 'float64',
    'Repaired?': 'bool'
}




def update_raw_tables(data_dir, ticker_list, sleep_time=30):
    '''
    Control flow
    - If ticker_name.csv not exists, download and store
    - If ticker_name.csv exists
        - If last date is not today, download and update
        - If last date is today, do nothing
    '''
    N = len(ticker_list)
    for i, ticker_name in enumerate(ticker_list):
        logging.info('%d/%d: %s'%(i, N, ticker_name))

        ticker = yf.Ticker(ticker_name)
        csv_pth = join(data_dir, f'{ticker_name}.csv')

        today = datetime.now().date()

        if os.path.exists(csv_pth):
            logging.info('Existing %s table found.'%(ticker_name))

            df = pd.read_csv(csv_pth, index_col=0)
            df = df.astype(RawDF_Schema)
            latest_datetime = df['Date'].max()  
            latest_day = latest_datetime.date() 
            endday = today - pd.Timedelta(days=1)
            if latest_day >= endday:
                logging.info(f'{ticker_name} is up to date. Last UTC date is {str(latest_day)} >= {endday}.')
            else:
                logging.info(f'{ticker_name} = {str(latest_day)}, while today is {today}')

                start_scrape_day = latest_day + pd.Timedelta(days=1)  # in DE time
                hist = ticker.history(
                    interval = '1d',
                    period = None,
                    end = today,  # exclusive, so the still-updating data today is not included.
                    start= start_scrape_day,  # Inclusive, so starting one day after the last day.
                    repair=True)
                hist.reset_index(inplace=True)
                hist = hist.astype(RawDF_Schema)
                logging.info('Extracted date %s to %s'%(str(hist['Date'].min().date()), str(hist['Date'].max().date())))
                df2 = pd.concat([df, hist], ignore_index=True)
                df2.to_csv(csv_pth)

        else:
            logging.info('Downloading %s for the whole period'%(ticker_name))
            hist = ticker.history(interval = '1d', period = None, end=today, repair=True)
            hist.reset_index(inplace=True)
            hist = hist.astype(RawDF_Schema)
            hist.to_csv(csv_pth)

        logging.debug('Sleeping for %d seconds'%(sleep_time))
        time.sleep(sleep_time)

if __name__ == '__main__':
    data_dir = 'data/yahoo/tickerdaily'
    ticker_list = load_ticker_list('yforseer/ticker_list.csv')
    update_raw_tables(data_dir, ticker_list)