import numpy as np
import pywt
import torch

def add_stockprice_noise(x, std, lam=0.1, A=1, return_tensor=False):
    """Add noise to the stock price data. See:
    Teng et al. 2020 (Enhancing Stock Price Trend Prediction via a Time-Sensitive Data Augmentation Method)


    Parameters
    ----------
    x : ndarray or torch.tensor
        Input signal with shape (t, ).

    std : float
    lam : float, optional
        Noise scaling factor dependent on index number, by default 0.1
    A : int, optional
        Noise scaling factor independent on index number, by default 1
    return_tensor : bool, optional
        If True, return torch.tensor. If False, return numpy.array. By default False.
    Returns
    -------
    ndarray or torch.tensor
        x array with added noise that is larger for higher index numbers.
    """


    ori_t = x.shape[0]

    # Decompose the signal
    coeffs = pywt.wavedec(x, 'db4', level=None)

    cA = coeffs[0]  # Approximation coefficients (low-frequency)
    cD = coeffs[1:]  # Detail coefficients (high-frequency)
    
    # Add noise for each component
    scale = std * A
    for i, cd in enumerate(cD):
        
        # Scale the noise term by the level of the detail coefficient
        # noise = (1-lam)^ik, eq(3) in Teng et al. 2020
        # noise is smaller when the index is lower (farther from the target)
        this_t = cd.shape[0]
        k = ori_t / this_t
        decay_power = np.arange(this_t) * k  
        decay_factor = (1 - lam) **decay_power * scale
        noise = np.random.normal(0, scale=decay_factor, size=this_t) 
        cD[i] = cd + noise

    # 4. Combine the modified high-frequency series with the low-frequency series
    modified_coeffs = [cA] + cD

    # 5. Reconstruct the x
    reconstructed_x = pywt.waverec(modified_coeffs, 'db4')
    if return_tensor:
        return torch.from_numpy(reconstructed_x)
    else:
        return reconstructed_x



class AddNoise:
    def __init__(self, stds, lam, A):
        """Add noise the a batch of time series data. The noise is larger for higher index numbers.

        Parameters
        ----------
        stds : numpy.ndarray
            Standard deviations for each time series in the batch. Shape (N, ) float32. Precomputed.
        lam : float
            Noise scaling factor dependent on index number.
        A : float
            Noise scaling factor independent on index number.
        """
        self.stds = stds 
        self.lam = lam
        self.A = A


    def __call__(self, x):
        """Transform x by adding noise to each time series in the batch.

        Parameters
        ----------
        x : torch.tensor
            Input batch of time series data. Shape (N, t) float32.

        Returns
        -------
        torch.tensor
            Transformed batch of time series data.
        """
        
        N = x.shape[0]
        x2 = x.clone()
        for i in range(N):
            std = self.stds[i]
            x2[i] = add_stockprice_noise(x[i], std, lam=self.lam, A=self.A, return_tensor=True)
        return x2


