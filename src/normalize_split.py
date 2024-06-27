import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def minmax_scale(x, mins, maxs):
    range_ = maxs - mins
    return (x - mins) / range_

def inverse_minmax_scale(x, mins, maxs):
    range_ = maxs - mins
    return x * range_ + mins




def normalize_split(input_data_pth: str, output_data_pth: str, output_stats_pth: str, test_size: float = 0.2):
    data = np.load(input_data_pth)
    X = data["X"]
    y = data['y']

    all = np.concatenate([X, y], axis=2)

    all2 = np.transpose(all, axes=(1, 0, 2))
    all3 = all2.reshape(all2.shape[0], -1)
    mins = all3.min(axis=1)
    maxs = all3.max(axis=1)

    # Normalize
    X_norm = minmax_scale(X, mins.reshape(1, -1, 1), maxs.reshape(1, -1, 1))
    y_norm = minmax_scale(y, mins.reshape(1, -1, 1), maxs.reshape(1, -1, 1))
    df_minmax = pd.DataFrame({'mins':mins, 'maxs':maxs})
    
    # Split train and test
    num_train = X_norm.shape[0] - int(X_norm.shape[0] * test_size)
    X_train, X_test = X_norm[:num_train, ...], X_norm[num_train:, ...]
    y_train, y_test = y_norm[:num_train, ...], y_norm[num_train:, ...]


    # Save data
    np.savez(output_data_pth, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    df_minmax.to_csv(output_stats_pth)


if __name__ == '__main__':

    input_data_pth = 'data/data_array.npz'
    output_data_pth = 'data/train_test_data.npz'
    output_stats_pth = 'data/minmax.csv'
    normalize_split(input_data_pth=input_data_pth, output_data_pth=output_data_pth, output_stats_pth=output_stats_pth)


