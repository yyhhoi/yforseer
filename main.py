import logging
from os.path import join
import datetime
from src.scrape import scrape_data
from src.convert_to_datarfame import convert_to_dataframe
from src.select_rows_columns import select_rows_columns
from src.select_X_y import select_X_y
from src.normalize_split import normalize_split
from src.train import train

def Kraken_Pipeline():

    # # ==========================================================
    # # Scrape data
    # # ==========================================================
    # scrape_data(input_file=join('primitives', 'crypto_list.txt'), 
    #             output_dir=join('data', 'kraken_scraped'), since=datetime(2022, 6, 15, 0, 0, 0), interval=1440)


    # # ==========================================================
    # # Convert to dataframe
    # # ==========================================================
    # convert_to_dataframe(input_data_dir= 'data/kraken_scraped', output_data_pth= 'data/kraken_allcurrencies.csv')


    # # ==========================================================
    # # Select rows and columns
    # # ==========================================================
    # select_rows_columns(input_data_pth='data/kraken_allcurrencies.csv',
    #                     output_data_pth='data/kraken_selected.csv')
    

    # # ==========================================================
    # # Select features and target
    # # ==========================================================
    # select_X_y(input_data_pth='data/kraken_selected.csv', memory=90, lookahead=30,
    #            output_data_pth='data/data_array.npz')
    
    # # ==========================================================
    # # Max-Min Normalization and train-test split
    # # ==========================================================
    # normalize_split(input_data_pth='data/data_array.npz', output_data_pth='data/train_test_data.npz',
    #                 output_stats_pth='minmax.csv', test_size=0.2)
    

    # ==========================================================
    # Train model and log metrics
    # ==========================================================
    train('data/train_test_data.npz', 'data/minmax.csv', 32, 20)
    
if __name__ == '__main__':
    Kraken_Pipeline()