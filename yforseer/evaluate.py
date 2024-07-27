import numpy as np
from numpy.typing import NDArray


def evaluate_stock_trend_prediction(xlast: NDArray, ypred: NDArray, ytest: NDArray, batch: bool= True):
    """Evaluate the stock trend prediction accuracy and return rate.

    Parameters
    ----------
    xlast : NDArray
        Last price of the stock. Shape (N, ) if batch is False else (M, N).
    ypred : NDArray
        Predicted price of the stock. Shape (N, ) if batch is False else (M, N).
    ytest : NDArray
        True price of the stock. Shape (N, ) if batch is False else (M, N).
    batch : bool, optional
        If True, calculate the metrics for all stocks in all samples of a batch. 
        If False, calculate the metrics for all stocks in one sample. 
        By default True.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray] if batch is True, else tuple[float, float, float]
        trend_accuracy: The accuracy that the trend (price rise or fall) is correctly predicted.
        buy_return: The return rate if the stocks were bought with $1 based on the predictions.
        sell_return: The return rate if the stocks were sold with $1 based on the predictions.
    """

    change_pred = (ypred - xlast) / xlast
    change_test = (ytest - xlast) / xlast

    # trend accuracy
    trend_pred = np.sign(change_pred)
    trend_test = np.sign(change_test)
    if batch:
        trend_accuracy = np.mean(trend_pred == trend_test, axis=1).squeeze()
        null_accuracy_rise = np.mean(trend_test > 0, axis=1).squeeze()
        null_accuracy_drop = np.mean(trend_test < 0, axis=1).squeeze()
    else:
        trend_accuracy = np.mean(trend_pred == trend_test)
        null_accuracy_rise = np.mean(trend_test > 0)
        null_accuracy_drop = np.mean(trend_test < 0)


    # Buy return rate
    change_pred_rise = change_pred.copy()
    change_test_rise = change_test.copy()
    fall_mask = change_pred < 0
    change_pred_rise[fall_mask] = 0
    change_test_rise[fall_mask] = 0
    if batch:
        buy_weights = change_pred_rise / change_pred_rise.sum(axis=1, keepdims=True)
        buy_return = (change_test_rise * buy_weights).sum(axis=1, keepdims=True).squeeze()
    else:
        buy_weights = change_pred_rise / change_pred_rise.sum()
        buy_return = (change_test_rise * buy_weights).sum()

    # Sell return rate
    change_pred_fall = change_pred.copy()
    change_test_fall = change_test.copy()
    rise_mask = change_pred > 0
    change_pred_fall[rise_mask] = 0
    change_test_fall[rise_mask] = 0
    if batch:
        sell_weights = np.abs(change_pred_fall) / np.abs(change_pred_fall).sum(axis=1, keepdims=True)
        sell_return = (-change_test_fall * sell_weights).sum(axis=1, keepdims=True).squeeze()
    else:
        sell_weights = np.abs(change_pred_fall) / np.abs(change_pred_fall).sum()
        sell_return = (-change_test_fall * sell_weights).sum()

    return (trend_accuracy, null_accuracy_rise, null_accuracy_drop), buy_return, sell_return


