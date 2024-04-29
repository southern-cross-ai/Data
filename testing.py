import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer
#test

# Importing custom modules
from model import Model
from dataset import English2FrenchDataset
from train import train
from config import TrainingConfig, ModelConfig

def main():
    # Tokenization
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    vocab_size = tokenizer.vocab_size
    if 0 < vocab_size:
        print(f'Tokenizer imported and vocab size is {vocab_size}')
    else:
        print(f'Tokenizer not imported: Vocab is {vocab_size}')

    # # Configuration instances
    # model_config = ModelConfig(vocab_size=vocab_size)
    # training_config = TrainingConfig()

    # # Data Preparation
    # dataset = English2FrenchDataset(training_config.csv_file, tokenizer, training_config.context_window)
    # dataloader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True)

    # # Model instantiation
    # model = Model(model_config)

    # # Loss and Optimizer
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)

    # # Assuming the use of a CUDA device if available
    # device = torch.device("cuda" if torch.cuda.is available() else "cpu")
    # model = model.to(device)
  
    # # Running the training loop
    # train(dataloader, model, loss_fn, optimizer, device)

if __name__ == "__main__":
    main()
