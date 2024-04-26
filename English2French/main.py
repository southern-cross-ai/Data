import pandas as pd
import math
from transformers import XLMRobertaTokenizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader

# get the data !wget https://raw.githubusercontent.com/southern-cross-ai/TranslationAI/main/English2French/eng_french.csv

# Constants
batch_size=2
context_window=512
embedding_size = 500
csv_colums=['English words/sentences', 'French words/sentences']

class English2FrenchDataset(Dataset):
    def __init__(self, csv_file, tokenizer, context_window):
        self.dataframe = pd.read_csv(csv_file, csv_colums)
        self.tokenizer = tokenizer
        self.context_window = context_window

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        source = self.dataframe.iloc[idx]['English words/sentences']
        target = self.dataframe.iloc[idx]['French words/sentences']

        source_encoded = self.tokenizer.encode_plus(
            source,
            context_window=self.context_window,
            padding='context_window',
            truncation=True,
            return_tensors='pt'
        )
        target_encoded = self.tokenizer.encode_plus(
            target,
            context_window=self.context_window,
            padding='context_window',
            truncation=True,
            return_tensors='pt'
        )

        # Generate masks for source and target
        source_mask = (source_encoded['input_ids'].squeeze() != self.tokenizer.pad_token_id)
        target_mask = (target_encoded['input_ids'].squeeze() != self.tokenizer.pad_token_id)

        return {
            'source_text': source,
            'source_input_ids': source_encoded['input_ids'].squeeze(),
            'source_mask': source_mask,
            'target_text': target,
            'target_input_ids': target_encoded['input_ids'].squeeze(),
            'target_mask': target_mask
        }

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, context_window=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(context_window).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * -(math.log(10000.0) / embedding_size))
        pe = torch.zeros(context_window, embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def main():
  # Tokenization
  tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base');
  vocab_size = tokenizer.vocab_size
  # Data Preparation
  dataset = English2FrenchDataset('eng_french.csv', tokenizer, context_window)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  # Testing
  iterator = iter(dataset)
  print(next(iterator)['source_text'])
  print(next(iterator)['target_text'])
  print(next(iterator)['source_input_ids'][0:10])
  print(next(iterator)['target_input_ids'][0:10])
  print(len(next(iterator)['source_input_ids']))
  print(len(next(iterator)['target_input_ids']))
  print(next(iterator)['target_mask'])

if __name__ == "__main__":
    main()
