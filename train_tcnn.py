# import yfinance as yf
import numpy as np
import torch
from torch.utils.data import DataLoader
from yforseer.datasets import StockDataset
from yforseer.trainers import StockTCNNTrainer
from yforseer.evaluate import evaluate_stock_trend_prediction
import mlflow
from tqdm import tqdm

dev = "cpu"
device = torch.device(dev)

batch_size = 32
epochs = 10
lr = 0.00001
test_frac = 0.1
memory = 60
lookahead = 30


# MLflow setup
remote_server_uri = "http://127.0.0.1:8080/"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("Stock last price prediction")
with mlflow.start_run():
    mlflow.log_params({'batch_size': batch_size, 'epochs': epochs, 'lr':lr, 'memory':memory, 'lookahead':lookahead,
                    'test_frac':test_frac})


    # Load data
    load_array_pth = 'data/yahoo/artifacts/data_array.npz'
    loaded_data = np.load(load_array_pth)
    data = torch.from_numpy(loaded_data['data']).to(torch.float32)
    mu, std = loaded_data['mu'], loaded_data['std']



    # Datasets and dataloaders
    num_days = data.shape[1]
    test_size = int(test_frac * num_days)
    train_size = num_days - test_size
    train_data = data[:, :train_size]
    test_data = data[:, train_size:]
    train_dataset = StockDataset(data = train_data, memory=memory, lookahead=lookahead, mode='last')
    test_dataset = StockDataset(data = test_data, memory=memory, lookahead=lookahead, mode='last')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    print('train_dataset:', len(train_dataset))
    print('test_dataset:', len(test_dataset))

    # Train
    num_inputs = 37
    num_channels = [64, 128, 256, 128, 64]
    lr = 0.0001
    trainer = StockTCNNTrainer(lr = lr, input_size=37, output_size=37, num_channels=num_channels, kernel_size=6, dropout=0.2)
    trainer.model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss_list, test_loss_list = [], []
        trainer.model.train()
        for X_train, y_train in (bar := tqdm(train_dataloader, leave=False)):
            train_loss, _ = trainer.train(X_train.to(device), y_train.to(device))
            train_loss_list.append(train_loss)
            bar.set_description(f'train_loss={train_loss:.4f}')
                                
        trainer.model.eval()
        all_x_test = []
        all_y_pred = []
        all_y_test = []
        for X_test, y_test in (bar := tqdm(test_dataloader, leave=False)):
            test_loss, y_pred = trainer.test(X_test.to(device), y_test.to(device))
            test_loss_list.append(test_loss)
            bar.set_description(f'test_loss={test_loss:.4f}')
            all_x_test.append(X_test.cpu().detach().numpy())
            all_y_pred.append(y_pred.cpu().detach().numpy().squeeze())
            all_y_test.append(y_test.cpu().detach().numpy().squeeze())

        all_x_test = np.concatenate(all_x_test, axis=0)
        all_y_pred = np.concatenate(all_y_pred, axis=0)
        all_y_test = np.concatenate(all_y_test, axis=0)
        all_x_test = all_x_test * std.reshape(1, -1, 1) + mu.reshape(1, -1, 1)
        all_y_pred = all_y_pred * std.reshape(1, -1) + mu.reshape(1, -1)
        all_y_test = all_y_test * std.reshape(1, -1) + mu.reshape(1, -1)
        (trend_acc, rise_acc, drop_acc), buy_returns, sell_returns = evaluate_stock_trend_prediction(all_x_test[:, :, -1], all_y_pred, all_y_test, batch=True)

        # Log prediction
        np.savez('predictions.npz', x_test=all_x_test, y_pred=all_y_pred, y_test=all_y_test)
        mlflow.log_artifact('predictions.npz')

        # Log metrics
        mlflow.log_metric('train_loss', np.mean(train_loss_list), step=epoch)
        mlflow.log_metric('test_loss', np.mean(test_loss_list), step=epoch)
        mlflow.log_metric('trend_accuracy_MD', np.median(trend_acc), step=epoch)
        mlflow.log_metric('trend_accuracy_LQ', np.quantile(trend_acc, 0.25), step=epoch)
        mlflow.log_metric('rise_accuracy_MD', np.median(rise_acc), step=epoch)
        mlflow.log_metric('drop_accuracy_MD', np.median(drop_acc), step=epoch)
        mlflow.log_metric('buy_return_MD', np.median(buy_returns), step=epoch)
        mlflow.log_metric('buy_return_LQ', np.quantile(buy_returns, 0.25), step=epoch)
        mlflow.log_metric('sell_return_MD', np.median(sell_returns), step=epoch)
        mlflow.log_metric('sell_return_LQ', np.quantile(sell_returns, 0.25), step=epoch)
    # Log model
    mlflow.pytorch.log_model(trainer.model, 'model')
