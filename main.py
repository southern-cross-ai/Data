import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer

# Importing custom modules
from model import Model
from dataset import TranslationDataset
from train import train
from config import TrainingConfig, ModelConfig

def main():
    # Tokenization
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    vocab_size = tokenizer.vocab_size
    # print(vocab_size)

    # # Configuration instances
    model_config = ModelConfig(vocab_size=vocab_size)
    training_config = TrainingConfig()

    # # Data Preparation
    dataset = TranslationDataset(training_config.csv_file, training_config.source_column, training_config.target_column, tokenizer, training_config.context_window)
    dataloader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True)
    data = next(iter(dataloader))
    print(data['source_input_ids'].shape)

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
    print("Working and done")

if __name__ == "__main__":
    main()

