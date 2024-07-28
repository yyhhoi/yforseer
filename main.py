from yforseer.scraping import update_raw_tables, load_ticker_list
from yforseer.prepare_data import combine_raw_csvs
from yforseer.prepare_data import determine_start_date, convert_to_array

def yforseer_pipeline():

    # # ==========================================================
    # # Scrape data
    # # ==========================================================
    
    # ticker_list = load_ticker_list('yforseer/ticker_list.csv')
    # update_raw_tables('data/yahoo/tickerdaily', ticker_list)

    # # ==========================================================
    # # Combine dataframes
    # # ==========================================================
    combine_raw_csvs(raw_data_dir = 'data/yahoo/tickerdaily', 
                     save_df_pth = 'data/yahoo/artifacts/combined.csv')
    
    # # ==========================================================
    # # Convert to array
    # # ==========================================================
    best_min_date = determine_start_date('data/yahoo/artifacts/combined.csv')
    convert_to_array('data/yahoo/artifacts/combined.csv', best_min_date, 'data/yahoo/artifacts/data_array.npz')




if __name__ == '__main__':
    yforseer_pipeline()