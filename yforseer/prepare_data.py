from glob import glob
from os.path import join
from .scraping import RawDF_Schema
import os
import pandas as pd
import logging

CombinedDF_Schema = {
    'ticker_id': 'int64',
    'ticker_code': 'string',
    'Date': 'datetime64[ns, Europe/Berlin]',
    'Close': 'float64'
}

def combine_df(raw_data_dir: str, save_df_pth: str) -> None:

    pths = glob(join(raw_data_dir, '*.csv'))
    all_dfs = []
    N_csv = len(pths)
    for i, pth in enumerate(pths):
        logging.debug(f'Loading {i}/{N_csv}: {pth}')

        
        # Load the raw data
        df = pd.read_csv(pth).astype(RawDF_Schema)

        # Extract the ticker code
        basefn = os.path.basename(pth)
        ticker_code = os.path.splitext(basefn)[0]
        
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
    