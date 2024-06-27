import json
import requests
from datetime import datetime
import time
from os.path import join
from tqdm import tqdm
import os
import logging

def scrape_data(input_file: str, output_dir: dir, since: datetime, interval:int = 1440, skip_exist=True):
    # Get the crypto currency names
    with open(input_file, mode='r') as f:
        foo = f.read()
    crypto_name_list = foo.replace('\n', '').split(',')


    unix_time = since.timestamp()
    var_time = datetime.fromtimestamp(unix_time)
    logging.info('Date time: %s' % str(var_time))
    logging.info("Unix time: %d" % unix_time)
    interval = 1440

    os.makedirs(output_dir, exist_ok=True)
    for crypto in tqdm(crypto_name_list):
        asset_pair = '%s/EUR'%(crypto)
        output_pth = join(output_dir, '%s.json'%(crypto))
        if os.path.exists(output_pth) and skip_exist:
            logging.info('%s exists. Skip.'%(output_pth))
            continue
        else:
            logging.info('%s not found. Scrapping.'%(output_pth))
            resp = requests.get(f'https://api.kraken.com/0/public/OHLC?pair={asset_pair}&interval={interval}&since={unix_time}')
            resp_str = json.dumps(resp.json())
            with open(output_pth, mode='w') as f:
                f.write(resp_str)

            time.sleep(6)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    input_file = join('primitives', 'crypto_list.txt')
    output_dir = join('data', 'kraken_scraped')
    since = datetime(2022, 6, 15, 0, 0, 0)
    interval = 1440

    scrape_data(input_file=input_file, output_dir=output_dir, since=since, interval=interval)