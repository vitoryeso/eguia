import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, learning_rate=1e-3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        self.dp1 = nn.Dropout(p=0.4)

        self.learning_rate = learning_rate

    def forward(self, x):
        # Propagando a entrada através da LSTM
        lstm_out, _ = self.lstm(x)
        # Passando a saída da LSTM para a camada linear
        x = self.fc(lstm_out[:, -1, :])
        x = self.dp1(x)
        predictions = self.fc2(F.relu(x))
        return predictions

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.MSELoss()(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        val_loss = nn.MSELoss()(y_pred, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        test_loss = nn.MSELoss()(y_pred, y)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

class TimeSeriesRNN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1, learning_rate=1e-3):
        super(TimeSeriesRNN, self).__init__()
        self.lr = learning_rate
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)  
        out = self.fc(out[:, -1, :])  # get the output of the last time step
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', val_loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)