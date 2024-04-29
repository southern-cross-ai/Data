from dataclasses import dataclass

@dataclass
class ModelConfig:
    embedding_size: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    vocab_size: int = None  # Initially set to None

@dataclass
class TrainingConfig:
    batch_size: int = 128
    context_window: int = 512
    csv_columns: list = ('English words/sentences', 'French words/sentences')

############# Example of file usage
# model_config = ModelConfig()  # Uses default values defined in the dataclass
# training_config = TrainingConfig()  # Uses default values defined in the dataclass

# You can still override defaults if necessary:
# custom_config = ModelConfig(embedding_size=256, dropout=0.2)
