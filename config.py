from dataclasses import dataclass

@dataclass
class ModelConfig:
    embedding_size: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    context_window: int = 512
    vocab_size: int = None  # Initially set to None

@dataclass
class TrainingConfig:
    batch_size: int = 4
    context_window: int = 512
    learning_rate: float = 0.0001
    source_column: str = 'English words/sentences' 
    target_column: str = 'French words/sentences'
    csv_file: str = 'data/eng_french.csv'

############# Example of file usage
# model_config = ModelConfig()  # Uses default values defined in the dataclass
# training_config = TrainingConfig()  # Uses default values defined in the dataclass

# You can still override defaults if necessary:
# custom_config = ModelConfig(embedding_size=256, dropout=0.2)
