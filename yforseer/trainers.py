from .networks import StockNet
import torch
import torch.nn as nn



class StockNetTrainer:
    def __init__(self, lr):
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.model = StockNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward_pass(self, X, y):
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        return loss, y_pred
    
    def test(self, X_test, y_test):
        with torch.no_grad():
            test_loss, y_pred = self.forward_pass(X_test, y_test)
        return test_loss.item(), y_pred
    
    def train(self, X_train, y_train):
        self.optimizer.zero_grad()
        train_loss, y_pred = self.forward_pass(X_train, y_train)
        train_loss.backward()
        self.optimizer.step()
        return train_loss.item(), y_pred