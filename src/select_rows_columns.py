import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta



def select_rows_columns(input_data_pth: str, output_data_pth: str, ref_name: str = 'BTC/EUR'):

    # Load data
    df = pd.read_csv(input_data_pth, index_col=0)

    # Choose reference currency for datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    ref_datetimes = df[df['id'] == ref_name]['datetime'].sort_values().reset_index(drop=True)

    # Select only the currencies with the exact datetimes
    names = df['id'].unique()
    selected_indexes = []
    admitted_names = []
    for name in names:
        sub_indexes = df.index[df['id'] == name]
        target_datetimes = df.loc[sub_indexes, 'datetime'].sort_values().reset_index(drop=True)

        if ref_datetimes.shape[0] != target_datetimes.shape[0]:
            continue
        compare = np.all(ref_datetimes == target_datetimes)
        if compare:
            admitted_names.append(name)
            selected_indexes.extend(sub_indexes)
    subdf = df.iloc[selected_indexes, :].loc[:, ['close', 'datetime', 'id']].reset_index(drop=True)

    # Output the file
    subdf.to_csv(output_data_pth)

if __name__ == '__main__':
    input_data_pth = 'data/kraken_allcurrencies.csv'
    output_data_pth = 'data/kraken_selected.csv'

    select_rows_columns(input_data_pth=input_data_pth,
                        output_data_pth=output_data_pth)