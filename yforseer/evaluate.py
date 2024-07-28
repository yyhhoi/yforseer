import numpy as np
from numpy.typing import NDArray

def weighted_return(change_pred: NDArray, change_test: NDArray, batch: bool=True) -> float | NDArray:
    """Calculate the return rate based on the predicted price change and the ground truth price change.
    Multiply "change_test" by -1 by you want to compute the sell return rate.

    Parameters
    ----------
    change_pred : NDArray
        Predicted price change in fraction. Shape (M, N) if batch is True else (N,).
    change_test : NDArray
        True price change in fraction. Shape (M, N) if batch is True else (N,).
    batch : bool, optional
        If True, calculate the metrics for all stocks in all samples of a batch. 
        If False, calculate the metrics for all stocks in one sample. 
        By default True.

    Returns
    -------
    float | NDArray
        The return rate. If batch is True, return shape is (M,) else float.
    """

    if batch:
        trade_weights = np.abs(change_pred) / (np.abs(change_pred).sum(axis=1, keepdims=True) + 1e-8)
        trade_return = (change_test * trade_weights).sum(axis=1).squeeze()
    else:
        trade_weights = np.abs(change_pred) / (np.abs(change_pred).sum() + 1e-8)
        trade_return = (change_test * trade_weights).sum()
    return trade_return


def compute_trend_accuracy(change_pred: NDArray, change_test: NDArray, batch: bool) -> float:
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

    return trend_accuracy, null_accuracy_rise, null_accuracy_drop

def compute_return_rate(change_pred: NDArray, change_test: NDArray, batch: bool, mode: str) \
    -> tuple[NDArray, NDArray]:
    """Compute the return rate based on the predicted price change and the ground truth price change.

    Parameters
    ----------
    change_pred : NDArray
        Predicted price change in fraction. Shape (M, N) if batch is True else (N,).
    change_test : NDArray
        True price change in fraction. Shape (M, N) if batch is True else (N,).
    batch : bool, optional
        If True, calculate the metrics for all stocks in all samples of a batch. 
        If False, calculate the metrics for all stocks in one sample. 
        By default True.
    mode : str
        'buy' or 'sell'. If 'buy', the return rate is calculated for buying stocks. If 'sell', the return rate is calculated for selling stocks.

    Returns
    -------
    tuple[NDArray, NDArray]:
        trade_returns: The return rate based on the predicted price change for top 1, 3, and all stocks. Shape (M, 3) if batch is True else (3,).
        perfect_trade_returns: The return rate with perfect prediction for top 1, 3, and all stocks. Shape (M, 3) if batch is True else (3,).

    Raises
    ------
    ValueError
        If mode is neither 'buy' nor 'sell'.
    """
    
    # Copy data
    change_pred_copied = change_pred.copy()
    change_test_copied = change_test.copy()


    # Buy or Sell mode
    if mode == 'buy':
        mask = change_pred <= 0  # Focus on buying. Exclude negative or null (zero-change) predictions
        multiplier = 1 
    elif mode == 'sell':
        mask = change_pred >= 0
        multiplier = -1
    else:
        raise ValueError("mode must be either 'buy' or 'sell'.")

    # Sort    
    change_pred_copied[mask] = 0
    change_test_copied[mask] = 0
    sorted_inds = np.argsort(multiplier*change_pred_copied, axis=-1)
    M = sorted_inds.shape[-1]

    trade_returns = []
    perfect_trade_returns = []
    for i in [1, 3, M]:
        
        # Select top 1, 3, all
        if batch:
            selected_tops_inds = sorted_inds[:, -i:]
            sorted_change_pred = np.take_along_axis(change_pred_copied, selected_tops_inds, axis=1)
            sorted_change_test = np.take_along_axis(change_test_copied, selected_tops_inds, axis=1)

        else:
            selected_tops_inds = sorted_inds[-i:]
            sorted_change_pred = change_pred_copied[selected_tops_inds]
            sorted_change_test = change_test_copied[selected_tops_inds]

        # Compute return based on prediction
        trade_return = weighted_return(sorted_change_pred, multiplier*sorted_change_test, batch=batch)

        # Compute return with perfect prediction
        perfect_trade_return = weighted_return(sorted_change_test, multiplier*sorted_change_test, batch=batch)

        # Store data
        trade_returns.append(trade_return)
        perfect_trade_returns.append(perfect_trade_return)

    if batch:
        trade_returns = np.stack(trade_returns, axis=1)  # Shape (M, 3)
        perfect_trade_returns = np.stack(perfect_trade_returns, axis=1)  # Shape (M, 3)
    else:
        trade_returns = np.array(trade_returns)  # Shape (3,)
        perfect_trade_returns = np.array(perfect_trade_returns)  # Shape (3,)

    return trade_returns, perfect_trade_returns

    


def evaluate_stock_trend_prediction(xlast: NDArray, ypred: NDArray, ytest: NDArray, batch: bool= True):
    """Evaluate the stock trend prediction accuracy and return rate.

    Parameters
    ----------
    xlast : NDArray
        Last price of the stock. Shape (M, N) if batch is True else (N,).
    ypred : NDArray
        Predicted price of the stock. Shape (M, N) if batch is True else (N,).
    ytest : NDArray
        True price of the stock. Shape (M, N) if batch is True else (N,).
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
    trend_accuracy, null_accuracy_rise, null_accuracy_drop = compute_trend_accuracy(change_pred, change_test, batch)

    # Buy return rate
    buy_returns, perfect_buy_returns = compute_return_rate(change_pred, change_test, batch, mode='buy')

    # Sell return rate
    sell_returns, perfect_sell_returns = compute_return_rate(change_pred, change_test, batch, mode='sell')

    return (trend_accuracy, null_accuracy_rise, null_accuracy_drop), (buy_returns, perfect_buy_returns), (sell_returns, perfect_sell_returns)


