from dataclasses import dataclass

@dataclass
class ModelConfig:
    embedding_size: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float

config = ModelConfig(embedding_size=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1)
