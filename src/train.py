import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .normalize_split import inverse_minmax_scale
import mlflow

class KrankenDataSet(Dataset):
    def __init__(self, input_data_pth, xlabel, ylabel):
        data = np.load(input_data_pth)
        self.X = torch.from_numpy(data[xlabel]).to(torch.float)
        self.y = torch.from_numpy(data[ylabel]).to(torch.float)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        
        return self.X[idx, ...], self.y[idx, ...]

class PriceNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=6, stride=2),  # -> (, 50, 43)
            nn.BatchNorm1d(num_features=50),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=100, kernel_size=7, stride=2),  # -> (, 100, 19)
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )


        self.flatten_and_transform = nn.Sequential(
            nn.Flatten(1, 2),  # -> (M*N, 1900)  
            nn.Linear(in_features=1900, out_features=200)  # -> (M*N, 200)
        )

        # -> (M, N, 200)
        self.attn_layer1 = nn.MultiheadAttention(
            embed_dim = 200,
            num_heads = 5,
            dropout = 0.2,
            batch_first=True,
        )  # -> (M, N, 200)

        self.linear_layer_1 = nn.Sequential(
            nn.Linear(in_features=200, out_features=200),
            nn.LayerNorm((183, 200)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        ) # -> (M, N, 200)

        self.attn_layer2 = nn.MultiheadAttention(
            embed_dim = 200,
            num_heads = 5,
            dropout = 0.2,
            batch_first=True,
        )  # -> (M, N, 200)


        self.linear_layer_final = nn.Sequential(
            nn.Linear(in_features=200, out_features=50),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=1),
        ) # -> (M, N, 1)

    def forward(self, X):
        # X: (M, N, time)
        M, N, _ = X.shape
        out = torch.flatten(X, 0, 1)  # (M, N, time) -> (M*N, time)
        out = torch.unsqueeze(out, dim=1)  # -> (M*N, 1, time)
        out = self.conv_layer1(out)  # -> (M*N, 50, 43)
        out = self.conv_layer2(out)  # -> (M*N, 100, 19)
        out = self.flatten_and_transform(out)  # -> (M*N, 200)
        out = out.reshape(M, N, out.shape[1])  # -> (M, N, 200)
        out, _ = self.attn_layer1(out, out, out, need_weights = False, average_attn_weights = False)  # -> (M, N, 200)
        out = self.linear_layer_1(out)
        out, _ = self.attn_layer2(out, out, out, need_weights = False, average_attn_weights = False)  # -> (M, N, 200)
        out = self.linear_layer_final(out)  # (M, N, 200) -> (M, N, 1)

        return out
    


class CryptoTrainer:
    def __init__(self, lr, minmax_pth=None):
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.model = PriceNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.mins = None
        self.maxs = None
        if minmax_pth:
            minmax_df = pd.read_csv(minmax_pth)
            self.mins = minmax_df['mins'].to_numpy().reshape(1, -1, 1)
            self.maxs = minmax_df['maxs'].to_numpy().reshape(1, -1, 1)


    def forward_pass(self, X, y):
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        return loss, y_pred
    
    def test(self, X_test, y_test):
        with torch.no_grad():
            test_loss, y_pred = self.forward_pass(X_test, y_test)
        return test_loss.item(), y_pred
    
    def train(self, X_train, y_train):
        train_loss, y_pred = self.forward_pass(X_train, y_train)
        train_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return train_loss.item()
    
    def eval(self, X_test, y_test):
        assert (self.mins is not None) and (self.maxs is not None)
        
        test_loss, y_pred = self.test(X_test, y_test)

        x_test_ori = inverse_minmax_scale(X_test[:, :, [-1]].numpy(), self.mins, self.maxs)
        y_test_ori = inverse_minmax_scale(y_test.numpy(), self.mins, self.maxs)
        y_pred_ori = inverse_minmax_scale(y_pred.numpy(), self.mins, self.maxs)
        
        # Compute fraction change
        pred_frac = (y_pred_ori - x_test_ori) / x_test_ori
        real_frac = (y_test_ori - x_test_ori) / x_test_ori
        weighting = np.abs(pred_frac)/np.sum(np.abs(pred_frac), axis=1).reshape(pred_frac.shape[0], 1, 1)
        aver_winrate = np.mean(np.sum(np.sign(pred_frac) * real_frac * weighting, axis=1))
        return test_loss, aver_winrate, (x_test_ori, y_test_ori, y_pred_ori)
        

def train(input_data_pth: str, input_minmax_pth: str, batch_size:int, epochs: int):

    # # Set up mlflow
    # Command: mlflow server --host 127.0.0.1 --port 8080
    remote_server_uri = "http://127.0.0.1:8080/"  # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("KrakenCrpytoRegression")
    mlflow.start_run(run_name='NoSeqAttend_NoRes_NoAvgpool_attn+1')
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('epochs', epochs)

    # # initialize datasets and dataloaders
    dataset_train = KrankenDataSet(input_data_pth, xlabel='X_train', ylabel='y_train')
    dataset_test = KrankenDataSet(input_data_pth, xlabel='X_test', ylabel='y_test')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))


    # Train
    trainer = CryptoTrainer(lr=1e-4, minmax_pth=input_minmax_pth)    
    for epoch in tqdm(range(epochs)):
        train_loss_list, test_loss_list = [], []
        trainer.model.train()
        for X_train, y_train in dataloader_train:
            train_loss = trainer.train(X_train, y_train)
            train_loss_list.append(train_loss)

        trainer.model.eval()
        for X_test, y_test in dataloader_test:
            pass
        test_loss, aver_winrate, _ = trainer.eval(X_test, y_test)

        # Log metrics
        mlflow.log_metric('train_loss', np.mean(train_loss_list), step=epoch)
        mlflow.log_metric('test_loss', test_loss, step=epoch)
        mlflow.log_metric('aver_winrate', aver_winrate, step=epoch)


    # Log model
    mlflow.pytorch.log_model(trainer.model, 'model')

    mlflow.end_run()

if __name__ == '__main__':
    # input_data_pth = 'data/train_test_data.npz'
    # input_minmax_pth = 'data/minmax.csv'
    # train(input_data_pth, input_minmax_pth, 32, 20)


    model = PriceNet()
    y = model(torch.rand(10, 183, 90))
    y.shape