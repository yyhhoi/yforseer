# import yfinance as yf
import numpy as np
import torch
from torch.utils.data import DataLoader
from yforseer.datasets import StockDataset
from yforseer.trainers import StockNetTrainer
import mlflow
from tqdm import tqdm


# MLflow setup
remote_server_uri = "http://127.0.0.1:8080/"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("Stock last price prediction")
mlflow.start_run()
mlflow.log_params({'batch_size': batch_size, 'epochs': epochs, 'lr':lr, 'NoiseGauSD':0.1})


# Load dataset
load_array_pth = 'data/yahoo/artifacts/data_array.npz'
data = torch.from_numpy(np.load(load_array_pth)['data']).to(torch.float32)
num_days = data.shape[1]
test_size = int(0.1 * num_days)
train_size = num_days - test_size
train_data = data[:, :train_size]
test_data = data[:, train_size:]
train_dataset = StockDataset(data = train_data, memory=60, lookahead=30, mode='last', noise=True)
test_dataset = StockDataset(data = test_data, memory=60, lookahead=30, mode='last', noise=False)
print('train_dataset:', len(train_dataset))
print('test_dataset:', len(test_dataset))


batch_size = 32
epochs = 20
lr = 0.0001



train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Train
trainer = StockNetTrainer(lr=1e-4)    
for epoch in tqdm(range(epochs)):
    train_loss_list, test_loss_list = [], []
    trainer.model.train()
    for X_train, y_train in (bar := tqdm(train_dataloader, leave=False)):
        train_loss, _ = trainer.train(X_train, y_train)
        train_loss_list.append(train_loss)
        bar.set_description(f'train_loss={train_loss:.4f}')
                            
    trainer.model.eval()
    for X_test, y_test in (bar := tqdm(test_dataloader, leave=False)):
        test_loss, _ = trainer.test(X_test, y_test)
        test_loss_list.append(test_loss)
        bar.set_description(f'test_loss={test_loss:.4f}')


    # Log metrics
    mlflow.log_metric('train_loss', np.mean(train_loss_list), step=epoch)
    mlflow.log_metric('test_loss', np.mean(test_loss_list), step=epoch)
    
# Log model
mlflow.pytorch.log_model(trainer.model, 'model')

mlflow.end_run()