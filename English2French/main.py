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

nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
max_seq_length = 100
dropout = 0.1

class English2FrenchDataset(Dataset):
    def __init__(self, csv_file, tokenizer, context_window):
        self.dataframe = pd.read_csv(csv_file, usecols=csv_colums)
        self.tokenizer = tokenizer
        self.context_window = context_window

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        source = self.dataframe.iloc[idx]['English words/sentences']
        target = self.dataframe.iloc[idx]['French words/sentences']

        source_encoded = self.tokenizer.encode_plus(
            source,
            max_length = context_window,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoded = self.tokenizer.encode_plus(
            target,
            max_length = context_window,
            padding='max_length',
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

class Model(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_size, 
                 nhead, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dim_feedforward, 
                 max_seq_length, 
                 dropout=0.1
                 ):
        super(Model, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout, max_seq_length)
        self.transformer = nn.Transformer(embedding_size, 
                                          nhead, 
                                          num_encoder_layers, 
                                          num_decoder_layers, 
                                          dim_feedforward, 
                                          dropout)
        self.out = nn.Linear(embedding_size, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        
        tgt = self.embedding(tgt) * math.sqrt(self.embedding_size)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(src, 
                                  tgt, 
                                  src_key_padding_mask=src_key_padding_mask, 
                                  tgt_mask=None,
                                  tgt_key_padding_mask=tgt_key_padding_mask, 
                                  memory_key_padding_mask=memory_key_padding_mask
                                  )
        output = self.out(output)
        return output


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
