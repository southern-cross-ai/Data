from dataclasses import dataclass

@dataclass
class ModelConfig:
    embedding_size: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    vocab_size: int = None  # Default to None, but intended to be set

def create_model_config(vocab_size: int):
    """ Factory function to create a ModelConfig with the specified vocab size. """
    return ModelConfig(vocab_size=vocab_size)

################ Example of file usage
# model_config = ModelConfig()  # Uses default values defined in the dataclass
# training_config = TrainingConfig()  # Uses default values defined in the dataclass

####################### You can still override defaults if necessary:
# custom_config = ModelConfig(embedding_size=256, dropout=0.2)
# config_instance = create_model_config(vocab_size=10000, dropout=0.2)
# config.dropout = 0.2 
