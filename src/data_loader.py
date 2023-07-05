import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, df, input_seq_length=50, label_seq_length=10, batch_size=64, workers=2):
        super().__init__()
        self.df = df
        self.input_seq_length = input_seq_length
        self.label_seq_length = label_seq_length
        self.batch_size = batch_size
        self.scaler = StandardScaler()

        self.num_workers = workers

    def setup(self, stage=None):
        # Primeiro, vamos dividir o DataFrame em recursos (X) e rótulos (y)
        X = self.df.drop(['valor'], axis=1).values
        y = self.df['valor'].values

        # Normalizando os dados
        X = self.scaler.fit_transform(X)
        y = self.scaler.fit_transform(y.reshape(-1, 1)).squeeze()

        # Convertendo para valores em float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Vamos dividir os dados em conjuntos de treinamento, validação e teste baseados no tempo
        train_ratio = 0.7
        valid_ratio = 0.15

        train_size = int(len(X) * train_ratio)
        valid_size = int(len(X) * valid_ratio)

        X_train, X_valid, X_test = X[:train_size], X[train_size:train_size + valid_size], X[train_size + valid_size:]
        y_train, y_valid, y_test = y[:train_size], y[train_size:train_size + valid_size], y[train_size + valid_size:]

        self.train_dataset = TimeSeriesDataset(X_train, y_train, self.input_seq_length, self.label_seq_length)
        self.valid_dataset = TimeSeriesDataset(X_valid, y_valid, self.input_seq_length, self.label_seq_length)
        self.test_dataset = TimeSeriesDataset(X_test, y_test, self.input_seq_length, self.label_seq_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, input_seq_length, label_seq_length):
        self.X = X
        self.y = y
        self.input_seq_length = input_seq_length
        self.label_seq_length = label_seq_length

    def __len__(self):
        return self.X.shape[0] - (self.input_seq_length + self.label_seq_length - 1)

    def __getitem__(self, idx):
        return (self.X[idx:idx+self.input_seq_length], self.y[idx+self.input_seq_length-1:idx+self.input_seq_length+self.label_seq_length-1])

def main():
    # Carregando o DataFrame (substitua 'your_data.csv' pelo caminho para o seu arquivo .csv)
    df = pd.read_csv('data/processed/processed_ordem_pagamento_producao.csv')

    # Criando uma instância do TimeSeriesDataModule
    data_module = TimeSeriesDataModule(df, input_seq_length=50, label_seq_length=10, batch_size=64)
    data_module.setup()

    # Preparando os dados (isso irá chamar o método setup)
    data_module.prepare_data()

    # Testando os dataloaders
    print("Testing train dataloader...")
    for X, y in data_module.train_dataloader():
        print("X shape: ", X.shape)
        print("y shape: ", y.shape)
        print("X type: " , type(X))
        print("X[0][0][0] type: ", type(X[0][0][0]))
        print("X[0][0][0]: ", X[0][0][0])

        break

    print("Testing validation dataloader...")
    for X, y in data_module.val_dataloader():
        print("X shape: ", X.shape)
        print("y shape: ", y.shape)
        break

    print("Testing test dataloader...")
    for X, y in data_module.test_dataloader():
        print("X shape: ", X.shape)
        print("y shape: ", y.shape)
        break

if __name__ == "__main__":
    main()