from glob import glob
from os.path import join
from .scraping import RawDF_Schema
import os
import pandas as pd
import numpy as np
import logging


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


def extract_features(combined_df_pth:str, save_npz_pth:str, memory:int = 30, lookahead:int = 30) -> None:
    """Extract features and target from the combined dataframe. It performs the following steps:
    1. Find the start datetime for a consistent time period across all tickers.
    2. Expand the dataframe rows to include all business days as Null.
    3. Interpolate missing values.
    4. Extract features (30 days price before).
    5. Extract targets (30th day price after, max, min, mean in the lookahead period).
    6. Normalize the prices for each ticker.

    Parameters
    ----------
    combined_df_pth : str
    
    save_npz_pth : str
    
    memory : int, optional
        Number of days to look back for prediction, by default 30.

    lookahead : int, optional
        Number of days to look forward for prediction, by default 30.

    """
    # Load data
    combined_df = pd.read_csv(combined_df_pth).astype(CombinedDF_Schema)
    
    # ============================
    # Find the start datetime
    # ============================
    max_datetime = combined_df['Date'].min()
    for ticker_code, ticker_df in combined_df.groupby('ticker_code'):
        ticker_mindate = ticker_df['Date'].min()
        if ticker_mindate > max_datetime:
            max_datetime = ticker_mindate
    start_datetime = max_datetime


    # ============================
    # Convert dataframe to arrays
    # ============================
    X_list, y_list = [], []
    total_end = combined_df['Date'].max()
    norm_table_dict = {'ticker_code':[], 'max_val':[], 'min_val':[]}
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
        print(f'{ticker_code} has {Nnan}/{Nrows} = {Nnan/Nrows:0.4f} missing values, filled with interpolation.')


        # Organize into array
        lpt = 0
        total_end = tmpdf.index[-1]
        x_ticker_tmp = []
        y_ticker_tmp = []
        while True:
            memory_end = lpt + memory
            target = memory_end + lookahead
            if target > (total_end):
                break

            # ================================
            # Extract features X and target y
            # ================================
            memory_mask = (tmpdf.index >= lpt) & (tmpdf.index < memory_end)
            lookahead_mask = (tmpdf.index >= memory_end) & (tmpdf.index <= target)
            x_tmp = tmpdf.loc[memory_mask, 'Close'].values.copy()  # (t, )
            y_tmp = tmpdf.iloc[target]['Close'].item()  # scalar
            lookahead_prices = tmpdf.loc[lookahead_mask, 'Close']
            ymax_tmp = lookahead_prices.max()  # scalar
            ymin_tmp = lookahead_prices.min()  # scalar
            ymu_tmp = lookahead_prices.mean()  # scalar

            x_ticker_tmp.append(x_tmp)
            y_ticker_tmp.append(np.array([y_tmp, ymax_tmp, ymin_tmp, ymu_tmp]))
            lpt += 1
            
        X_ticker = np.stack(x_ticker_tmp, axis=0)  # (n, t)
        y_ticker = np.stack(y_ticker_tmp, axis=0)  # (n, 4)

        # ================================
        # Normalize
        # ================================
        max_val = tmpdf['Close'].max()
        min_val = tmpdf['Close'].min()
        X_ticker = (X_ticker - min_val) / (max_val - min_val)
        y_ticker = (y_ticker - min_val) / (max_val - min_val)
        norm_table_dict['ticker_code'].append(ticker_code)
        norm_table_dict['max_val'].append(max_val)
        norm_table_dict['min_val'].append(min_val)

        X_list.append(X_ticker)
        y_list.append(y_ticker)

    # Stack the arrays
    Xtmp = np.stack(X_list)  # (NumNames, M, t)
    ytmp = np.stack(y_list)  # (NumNames, M, 4)
    X = np.transpose(Xtmp, (1, 0, 2))  # (M, NumNames, t,)
    y = np.transpose(ytmp, (1, 0, 2))  # (M, NumNames, 4,)
    norm_table = pd.DataFrame(norm_table_dict)

    print("X's shape = \n%s" % str(X.shape))
    print("y's shape = \n%s" % str(y.shape))

    # Save the data
    np.savez(save_npz_pth, X=X, y=y)
    save_norm_pth = join(os.path.splitext(save_npz_pth)[0] + '_norm.csv')
    norm_table.to_csv(save_norm_pth, index=False)