# import yfinance as yf
import numpy as np
import torch
from torch.utils.data import DataLoader
from yforseer.datasets import StockDataset
from yforseer.trainers import StockNetTrainer
from yforseer.augmentation import AddNoise
import mlflow
from tqdm import tqdm


batch_size = 32
epochs = 20
lr = 0.0001
noise_lam = 0.05
noise_A = 0.5
test_frac = 0.1
memory = 60
lookahead = 30


# MLflow setup
remote_server_uri = "http://127.0.0.1:8080/"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("Stock last price prediction")
with mlflow.start_run():
    mlflow.log_params({'batch_size': batch_size, 'epochs': epochs, 'lr':lr, 'memory':memory, 'lookahead':lookahead,
                    'test_frac':test_frac, 'noise_lam':noise_lam, 'noise_A':noise_A})


    # Load data
    load_array_pth = 'data/yahoo/artifacts/data_array.npz'
    data = torch.from_numpy(np.load(load_array_pth)['data']).to(torch.float32)

    # Augmentation function
    stds = torch.diff(data, dim=1).std(dim=1).numpy()
    addnoise_transform = AddNoise(stds, noise_lam, noise_A)

    # Datasets and dataloaders
    num_days = data.shape[1]
    test_size = int(test_frac * num_days)
    train_size = num_days - test_size
    train_data = data[:, :train_size]
    test_data = data[:, train_size:]
    train_dataset = StockDataset(data = train_data, memory=memory, lookahead=lookahead, mode='last', transform=addnoise_transform)
    test_dataset = StockDataset(data = test_data, memory=memory, lookahead=lookahead, mode='last')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    print('train_dataset:', len(train_dataset))
    print('test_dataset:', len(test_dataset))

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
