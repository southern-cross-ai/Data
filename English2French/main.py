import pandas as pd
import math
from transformers import XLMRobertaTokenizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Constants
batch_size=128
context_window=512
embedding_size = 512
csv_columns=['English words/sentences', 'French words/sentences']

nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1


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

def train(dataloader, model, loss_fn, optimizer, device):
    print('Training...')
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['source_input_ids'].to(device)
        print(src)
        tgt = batch['target_input_ids'].to(device)
        src_mask = batch['source_mask'].to(device)
        tgt_mask = batch['target_mask'].to(device)

        # Transpose masks to match the expected shape [seq_length, batch_size]
        src_mask = src_mask.t()
        tgt_mask = tgt_mask.t()

        optimizer.zero_grad()
        
        output = model(src, tgt, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        # print(output)
        output = output.reshape(-1, output.shape[-1])  # Flatten output for loss calculation
        tgt = tgt.reshape(-1)  # Flatten target for loss calculation

        loss = loss_fn(output, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print (total_loss)

    average_loss = total_loss / len(dataloader)
    print(f"Average Loss: {average_loss}")

    # Assuming the use of a CUDA device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Example of running the training loop
    train(dataloader, model, loss_fn, optimizer)

  
if __name__ == "__main__":
    main()
