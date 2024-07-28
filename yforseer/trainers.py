from .networks import StockNet, StockTCNN
import torch
import torch.nn as nn

class Trainer:
    def __init__(self, lr, model):
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward_pass(self, X, y):
        y_pred = self.model(X)
        # breakpoint()
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
    
    def save(self, pth, epoch):
        checkpoint = { 
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr': self.lr
            }
        torch.save(checkpoint, pth)

    def load(self, pth):
        checkpoint = torch.load(pth)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr = checkpoint['lr']



class StockNetTrainer(Trainer):
    def __init__(self, lr):
        model = StockNet()
        super().__init__(lr, model)


class StockTCNNTrainer(Trainer):
    def __init__(self, lr, input_size, output_size, num_channels, kernel_size, dropout):
        model = StockTCNN(input_size, output_size, num_channels, kernel_size, dropout)
        super().__init__(lr, model)