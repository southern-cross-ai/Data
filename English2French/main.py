import pandas as pd
import math
from transformers import XLMRobertaTokenizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def main():
  # Tokenization
  tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
  vocab_size = tokenizer.vocab_size
  # Data Preparation
  dataset = English2FrenchDataset('eng_french.csv', tokenizer, context_window)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  # Model
  model = Model(vocab_size, embedding_size, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, context_window, dropout)
  # Loss
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001)

  # Assuming the use of a CUDA device if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  
  # Example of running the training loop
  train(dataloader, model, loss_fn, optimizer)

  
if __name__ == "__main__":
    main()
