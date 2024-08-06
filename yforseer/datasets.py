import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data, memory, lookahead, mode, transform=None):
        self.data = data  # (N_tickers, N_days), torch.tensor, float32
        self.memory = memory
        self.lookahead = lookahead
        if mode not in ['last', 'all', 'stats']:
            raise ValueError('Invalid mode from StockDataset constructor. Choose from "last", "all" or "stats".')
        self.mode = mode  # last/all/stats
        self.transform = transform
        self.window = memory + lookahead
        self.n_tickers = self.data.shape[0]
        self.n_days = self.data.shape[1]
        self.final_start_ind = self.n_days - self.window
        self.n_samples = self.final_start_ind + 1


        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        
        start, mid, end = idx, idx+self.memory, idx+self.window
        if end > self.n_days:
            raise IndexError('Index out of bounds')
        
        # Features
        X = self.data[:, start:mid]
        if self.transform:
            X = self.transform(X)
        

        # Target
        target = None
        y_interval = self.data[:, mid:end]
        if self.mode == 'all':
            target = y_interval
        elif self.mode == 'last':
            target = y_interval[:, [-1]]
        elif self.mode == 'stats':
            ylast = y_interval[:, [-1]]
            ymin = y_interval.min(dim=1, keepdim=True)[0]   
            ymax = y_interval.max(dim=1, keepdim=True)[0]
            target = torch.cat([ylast, ymin, ymax], dim=1)
        else:
            raise ValueError('Invalid mode from StockDataset constructor. Choose from "last", "all" or "stats".')
        return X, target


class StockDiffDataset(StockDataset):
    def __init__(self, data, memory, lookahead=1, mode='last', return_price=False):
        super().__init__(data=data, memory=memory, lookahead=lookahead, mode=mode, transform=None)
        self.return_price = return_price


    def __getitem__(self, idx):
        
        start, mid, end = idx, idx+self.memory, idx+self.window
        if end > self.n_days:
            raise IndexError('Index out of bounds')
        
        # Features
        prices = self.data[:, start:end]  # (N_tickers, window)
        Xprices = self.data[:, start:mid]
        
        if self.mode == 'last':
            yprices = self.data[:, [end-1]]
        elif self.mode == 'all':
            yprices = self.data[:, mid:end]
        else:
            raise ValueError('Invalid mode for StockDiffDataset. Choose from "last" or "all".')


        # Difference
        Xdiff = torch.diff(Xprices, dim=1)
        xlast = Xprices[:, [-1]]
        ydiff_0 = yprices[:, [0]] - xlast
        if (self.lookahead > 1):
            ydiff_rest = torch.diff(yprices, dim=1)
            ydiff = torch.cat([ydiff_0, ydiff_rest], dim=1)
        else:
            ydiff = ydiff_0



        if self.return_price:
            return Xdiff, ydiff, Xprices, yprices, prices
        else:
            return Xdiff, ydiff, Xprices, yprices