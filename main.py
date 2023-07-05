import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.models import LSTMRegressor, TimeSeriesRNN
from src.data_loader import TimeSeriesDataModule
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def main():
    # Defina os parâmetros do modelo
    INPUT_DIM = 377
    HIDDEN_DIM = 5
    OUTPUT_DIM = 1
    NUM_LAYERS = 1
    LEARNING_RATE = 1e-3

    # Criar instância do SummaryWriter
    tb_writer = SummaryWriter('logs')


    # Carregando o DataFrame (substitua 'your_data.csv' pelo caminho para o seu arquivo .csv)
    df = pd.read_csv('data/processed/processed_ordem_pagamento_producao.csv')

    # Criando uma instância do TimeSeriesDataModule
    data_module = TimeSeriesDataModule(df, input_seq_length=6, label_seq_length=1, batch_size=64, workers=8)
    data_module.setup()

    # Preparando os dados (isso irá chamar o método setup)
    data_module.prepare_data()

    # Inicialize o modelo
    #model = LSTMRegressor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, LEARNING_RATE)
    model = TimeSeriesRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, LEARNING_RATE)

    # Inicialize o treinador (substitua 'tb_logs' pelo caminho para o seu diretório de logs)
    trainer = Trainer(max_epochs=100, logger=pl.loggers.TensorBoardLogger('tb_logs', name='my_model'), 
                        callbacks=[pl.callbacks.ModelCheckpoint(dirpath='checkpoints')], 
                        default_root_dir='.')

    # Treine o modelo
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    # Teste o modelo
    trainer.test(model, data_module.test_dataloader())

    # Feche o SummaryWriter
    tb_writer.close()

if __name__ == '__main__':
    main()
