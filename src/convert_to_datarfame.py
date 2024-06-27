import glob
import os
from os.path import join
import json
import pandas as pd
import logging
from tqdm import tqdm
from datetime import datetime

def convert_to_dataframe(input_data_dir: str, output_data_pth:str):

    pths= glob.glob(join(input_data_dir, '*'))
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volumne', 'count']
    df_list = []
    for pth in tqdm(pths):
        with open(pth, mode='r') as f:
            data = json.load(f)
        crypto_name = os.path.splitext(os.path.basename(pth))[0]
        crypto_id = '%s/EUR'%(crypto_name)
        df = pd.DataFrame(data['result'][crypto_id], columns=columns)
        df['datetime'] = df['timestamp'].apply( lambda x : datetime.fromtimestamp(x))
        df = df.astype({
            'timestamp': 'int64',
            'open': float ,
            'high': float ,
            'low': float ,
            'close': float ,
            'vwap': float ,
            'volumne': float ,
            'count': 'int64',
            'datetime': 'datetime64[ns]'  
        })
        df['id'] = crypto_id
        df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True)
    df_all.to_csv(output_data_pth)


if __name__ == "__main__":
    input_data_dir = 'data/kraken_scraped'
    output_data_pth = 'data/kraken_allcurrencies.csv'
    convert_to_dataframe(input_data_dir, output_data_pth)
