from glob import glob
from os.path import join
from .scraping import RawDF_Schema
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from .utils import print_title

CombinedDF_Schema = {
    'ticker_id': 'int64',
    'ticker_code': 'string',
    'Date': 'datetime64[ns, Europe/Berlin]',
    'Close': 'float64'
}

def combine_raw_csvs(raw_data_dir: str, save_df_pth: str) -> None:
    """Combine raw data from multiple csv files into a single dataframe.

    Parameters
    ----------
    raw_data_dir : str

    save_df_pth : str
    """
    pths = glob(join(raw_data_dir, '*.csv'))
    all_dfs = []
    N_csv = len(pths)
    for i, pth in enumerate(pths):
        logging.debug(f'Loading {i}/{N_csv}: {pth}')

        # Extract the ticker code
        basefn = os.path.basename(pth)
        ticker_code = os.path.splitext(basefn)[0]

        # Load the raw data
        df = pd.read_csv(pth).astype(RawDF_Schema).sort_values(by='Date').reset_index(drop=True)
        
        # Select useful columns. 
        # Append the ticker code and ticker id to the dataframe
        df_to_append = df[['Date', 'Close']].copy()
        df_to_append['ticker_id'] = i
        df_to_append['ticker_code'] = ticker_code
        all_dfs.append(df_to_append)

    # Combine the dataframe
    logging.debug(f'Combining dataframe to {save_df_pth}')
    combined_df = pd.concat(all_dfs, ignore_index=True)
    reordered_cols = ['ticker_id', 'ticker_code', 'Date', 'Close']

    # Save the dataframe
    logging.debug(f'Saving combined dataframe to {save_df_pth}')
    combined_df[reordered_cols].to_csv(save_df_pth, index=False)


def determine_start_date(combined_df_pth: str) -> pd.Timestamp:
    '''
    - Find the start datetime for a consistent time period across all tickers.
    '''
    print_title('Determine the best min date ')
    # Load data
    combined_df = pd.read_csv(combined_df_pth).astype(CombinedDF_Schema)
    
    # Min dates for tickers
    min_dates = combined_df.groupby('ticker_code').apply(lambda x: x['Date'].min())
    min_dates.sort_values(inplace=True)

    # Precalculate all business days
    start_date = min_dates.min()
    end_date = pd.Timestamp(datetime.now(), tz='Europe/Berlin')
    all_datetimes = pd.date_range(start=start_date, end=end_date, freq='B')

    # Compute the number of data points for each min date
    data_points = []
    for mindate in min_dates:
        N = (min_dates <= mindate).sum()
        D = (all_datetimes >= mindate).sum()
        data_points.append(N*D)

    # Best min date with the most data points
    max_ind = np.argmax(data_points)
    best_min_date = min_dates.iloc[max_ind]
    N = (min_dates <= best_min_date).sum()

    # Check the results
    print(pd.DataFrame({'min_dates': min_dates, 'data_points': data_points}))
    print(f'Best min date = {best_min_date}')
    print('Included tickers = %d' % (N))
    return best_min_date

def convert_to_array(combined_df_pth:str, start_datetime: pd.Timestamp,
                     save_npz_pth: str) -> None:
    """Convert the combined dataframe to a numpy array. It performs the following steps:


    1. Expand the dataframe rows to include all business days as Null.
    2. Interpolate missing values.
    3. Normalize the prices for each ticker.

    Parameters
    ----------
    combined_df_pth : str

    start_datetime : pd.Timestamp

    save_npz_pth : str

    """

    print_title('Convert to array')

    # Load data
    combined_df = pd.read_csv(combined_df_pth).astype(CombinedDF_Schema)

    # Loop for each ticker
    ticker_array= []
    all_ticker_codes = []
    for ticker_code, ticker_df in combined_df.groupby('ticker_code'):

        # Slice the dataframe from the start datetime
        if ticker_df['Date'].min() > start_datetime:
            continue
        mask = ticker_df['Date'] >= start_datetime
        subperiod_df = ticker_df[mask]

        # ==========================================
        # Interpolate missing values (business days)
        # ==========================================
        tmpdf = subperiod_df.set_index('Date').asfreq('B')
        Nnan = tmpdf['Close'].isna().sum()
        Nrows = len(tmpdf)
        tmpdf['ticker_id'] = tmpdf['ticker_id'].ffill().astype(int)
        tmpdf['ticker_code'] = tmpdf['ticker_code'].ffill().astype(str)
        tmpdf['Close'] = tmpdf['Close'].interpolate(method='linear')
        tmpdf.reset_index(inplace=True)
        ticker_array.append(tmpdf['Close'].values)
        all_ticker_codes.append(ticker_code)
        print(f'{ticker_code} has {Nnan}/{Nrows} = {Nnan/Nrows:0.4f} missing values, filled with interpolation.')
    
    # Stack the arrays
    ticker_array = np.stack(ticker_array)

    # ================================
    # Normalize
    # ================================
    mu = ticker_array.mean(axis=1)
    std = ticker_array.std(axis=1)
    ticker_array = (ticker_array - mu.reshape(-1, 1)) / std.reshape(-1, 1)
    print('Ticker array shape = %s' % str(ticker_array.shape))

    # ================================
    # Save data
    # ================================
    np.savez(save_npz_pth, data=ticker_array, mu=mu, std=std, ticker_codes=all_ticker_codes)
    print('Array saved at %s' % save_npz_pth)

