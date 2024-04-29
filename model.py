import torch
import torch.nn as nn
import math

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
                 context_window, 
                 dropout=0.1
                 ):
        super(Model, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout, context_window)
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
