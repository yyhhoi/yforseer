import torch
import torch.nn as nn

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
        out, _ = self.attn_layer1(out, out, out, need_weights = False, average_attn_weights = False)  # -> (M, N, 200)
        out = self.linear_layer_1(out)
        out, _ = self.attn_layer2(out, out, out, need_weights = False, average_attn_weights = False)  # -> (M, N, 200)
        out = self.linear_layer_final(out)  # (M, N, 200) -> (M, N, 1)

        return out