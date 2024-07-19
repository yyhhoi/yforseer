import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data, memory, lookahead):
        self.data = data  # (N_tickers, N_days), torch.tensor, float32
        self.memory = memory
        self.lookahead = lookahead
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
        X = self.data[:, start:mid]
        y_interval = self.data[:, mid:end]
        ylast = y_interval[:, [-1]]
        ymin = y_interval.min(dim=1, keepdim=True)[0]   
        ymax = y_interval.max(dim=1, keepdim=True)[0]   
        ymean = y_interval.mean(dim=1, keepdim=True)
        y = torch.cat([ylast, ymin, ymax, ymean], dim=1)
        return X, y
