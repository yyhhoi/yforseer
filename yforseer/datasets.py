import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data, memory, lookahead, mode, noise=False):
        self.data = data  # (N_tickers, N_days), torch.tensor, float32
        self.memory = memory
        self.lookahead = lookahead
        self.mode = mode  # last/all/stats
        self.noise = noise
        self.window = memory + lookahead
        self.n_tickers = self.data.shape[0]
        self.n_days = self.data.shape[1]
        self.final_start_ind = self.n_days - self.window
        self.n_samples = self.final_start_ind + 1

        # Noise
        self.stds = torch.std(self.data, dim=1).reshape(-1, 1) / 10

        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        
        start, mid, end = idx, idx+self.memory, idx+self.window
        if end > self.n_days:
            raise IndexError('Index out of bounds')
        
        X = self.data[:, start:mid]
        if self.noise:
            X += torch.normal(mean=0, std=self.stds)
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
