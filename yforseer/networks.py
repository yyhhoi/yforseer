import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class StockNet(nn.Module):
    def __init__(self, final_dim=1):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=6, stride=2),  # (, 1, 60)-> (, 100, 28)
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=200, kernel_size=6, stride=2),  # -> (, 200, 12)
            nn.BatchNorm1d(num_features=200),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=200, out_channels=400, kernel_size=6, stride=2),  # -> (, 400, 4)
            nn.BatchNorm1d(num_features=400),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.flatten_and_transform = nn.Sequential(
            nn.Flatten(1, 2),  # -> (M*N, 1600)  
            nn.Linear(in_features=1600, out_features=200)  # -> (M*N, 200)
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
            nn.LayerNorm((37, 200)),
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
            nn.LayerNorm((37, 50)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=final_dim),
        ) # -> (M, N, 1)

    def forward(self, X):
        # X: (M, N, time)
        M, N, _ = X.shape
        out = torch.flatten(X, 0, 1)  # (M, N, time) -> (M*N, time)
        out = torch.unsqueeze(out, dim=1)  # -> (M*N, 1, time)
        out = self.conv_layer1(out)  # 
        out = self.conv_layer2(out)  # 
        out = self.conv_layer3(out)  # -> (M*N, 400, 4)
        
        out = self.flatten_and_transform(out)  # -> (M*N, 200)
        out = out.reshape(M, N, out.shape[1])  # -> (M, N, 200)
        out_r, _ = self.attn_layer1(out, out, out, need_weights = False, average_attn_weights = False)  # -> (M, N, 200)
        out = self.linear_layer_1(out + out_r)
        out_r, _ = self.attn_layer2(out, out, out, need_weights = False, average_attn_weights = False)  # -> (M, N, 200)
        out = self.linear_layer_final(out + out_r)  # (M, N, 200) -> (M, N, 1)
        out = out + X[:, :, [-1]]

        return out
    


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    


class StockTCNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(StockTCNN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels[-1], output_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2, 1)
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.01)
        self.linear2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.tcn(x)
        out = self.linear1(out[:, :, -1]).unsqueeze(2)
        out = self.relu(torch.cat([out, x[:, :, [-1]]], dim=2))
        out = self.linear2(out)
        return out