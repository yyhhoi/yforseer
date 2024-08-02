
def print_title(title: str, width: int = 80):
    print("=" * width)
    print(title.center(width))
    print("=" * width)


calc_kernel = lambda x, k, s:  (x-k)//s + 1


def convert_frac_to_prices(ypred, prices):
    """Converts the fractional change in stock price to the actual stock price.

    Parameters
    ----------
    ypred : NDArray
        (Predicted) fractional change in stock price. Shape = (M, N, 1).
    prices : NDArray
        Stock prices which were used to generate the fractional changes data X and y. (M, N, T)

    Returns
    -------
    NDArray, NDArray
        Prices data X and the predicted prices y.
    """
    X_prices = prices[:, :, :-1]
    X_last = X_prices[:, :, [-1]]  # Same as directly indexing prices[:, [-2]]
    y_price_pred = ypred * X_last + X_last
    return X_prices, y_price_pred


def convert_diff_to_prices(ypred, prices):
    """Converts differences in stock price to the actual stock price.

    Parameters
    ----------
    ypred : NDArray
        (Predicted) fractional change in stock price. Shape = (M, N, 1).
    prices : NDArray
        Stock prices which were used to generate the fractional changes data X and y. (M, N, T)

    Returns
    -------
    NDArray, NDArray
        Prices data X and the predicted prices y.
    """
    X_prices = prices[:, :, :-1]
    X_last = X_prices[:, :, [-1]]  # Same as directly indexing prices[:, [-2]]
    y_price_pred = ypred + X_last
    return X_prices, y_price_pred

