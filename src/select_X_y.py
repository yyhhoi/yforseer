import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import logging



def select_X_y(input_data_pth: str, memory: int, lookahead: int, output_data_pth):
    logging.info('Start function - select_X_y')

    logging.info('loading dataframe')
    df = pd.read_csv(input_data_pth, index_col=0)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].apply(lambda x : x.date())

    names = df['id'].unique()
    ref_datetimes = df.loc[df['id'] == 'BTC/EUR', 'date']
    total_end = ref_datetimes.max()
    x_list = []
    y_list = []
    logging.info('Looping for each currency')
    for name in tqdm(names):
        
        sub_indexes = df.index[df['id'] == name]

        sub_datetimes = df.iloc[sub_indexes]['date']
        x_name_tmp = []
        y_name_tmp = []
        lpt = ref_datetimes.min()
        while True:
            memory_end = lpt + timedelta(days = memory)
            target = memory_end + timedelta(days = lookahead)
            if target > (total_end):
                break

            features_dt_index = sub_indexes[(sub_datetimes >= lpt) & (sub_datetimes < memory_end)]
            target_dt_index = sub_indexes[sub_datetimes == target]
            df_tmp = df.iloc[features_dt_index].sort_values(by='datetime')
            x_tmp = df_tmp['close'].values.copy()  # (t, )
            y_tmp = df.iloc[target_dt_index]['close'].values.copy()  # (1, )
            assert y_tmp.shape[0] == 1

            x_name_tmp.append(x_tmp)
            y_name_tmp.append(y_tmp)
            lpt += timedelta(days = 1)

            
        x_list.append(np.stack(x_name_tmp))  # (M, t, )
        y_list.append(np.stack(y_name_tmp))  # (M, 1, )

    Xtmp = np.stack(x_list)  # (NumNames, M, t,)
    ytmp = np.stack(y_list)  # (NumNames, M, 1, )

    X = np.transpose(Xtmp, (1, 0, 2))  # (M, NumNames, t,)
    y = np.transpose(ytmp, (1, 0, 2))  # (M, NumNames, 1,)

    logging.info("X's shape = \n%s" % str(X.shape))
    logging.info("y's shape = \n%s" % str(y.shape))
    logging.info('Saving data at %s' % output_data_pth)
    np.savez(output_data_pth, X=X, y=y)



if __name__ == '__main__':
    input_data_pth = 'data/kraken_selected.csv'
    memory = 90  # in days
    lookahead = 30  # in days
    output_data_pth = 'data/data_array.npz'
    select_X_y(input_data_pth=input_data_pth, memory=memory, lookahead=lookahead,
               output_data_pth=output_data_pth)