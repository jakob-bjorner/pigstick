"""
Basic config for SFT.
"""

from typing import Optional
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """
    Config for the model. 
    """
    # Model hyperparameters
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth"
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model"

class TokenizerConfig(BaseModel):
    """
    Config for the tokenizer. 
    """
    type: str = "sentencepiece"
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model"
    assert type in ["sentencepiece", "huggingface"], "Tokenizer type must be either sentencepiece or huggingface"


class DataConfig(BaseModel):
    """
    Config for the data. 
    """
    data_dir: str = "data/alpaca"
    out_dir: str = "out/full/alpaca"

class TrainingLoopConfig(BaseModel):
    """ 
    Config for the training loop. 
    """
    instruction_tuning: bool = True
    eval_interval: int = 1000
    save_interval: int = 1000
    eval_iters: int = 100
    log_interval: int = 100
    logging: bool = True

    # Training loop hyperparameters
    learning_rate: float = 3e-5
    batch_size: int = 128
    micro_batch_size: int = 8
    epoch_size: int = 30000
    num_epochs: int = 3
    weight_decay: float = 0.0
    block_size: int = 512
    warmup_iters: int = 100

class Config(BaseModel):
    """ 
    Main config for SFT. 
    """
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    training: TrainingLoopConfig = Field(default_factory=TrainingLoopConfig)
    seed: int = 42

    @property
    def gradient_accumulation_iters(self) -> int:
        return self.model.batch_size // self.model.micro_batch_size

    @property
    def max_iters(self) -> int:
        return self.model.num_epochs * (self.model.epoch_size // self.model.micro_batch_size)

        



def load_config(config_path: str = None) -> Config:
    """ 
    Load config from a yaml file. 
    """
    import yaml
    if config_path is None:
        print("No config path provided, using default config")
        return Config()
    with open(config_path, 'r') as f:   
        print(f"Loading config from {config_path}")
        config_dict = yaml.safe_load(f)
    print("Loaded config:")
    print(yaml.dump(config_dict, default_flow_style=False))
    return Config.model_validate(config_dict)



if __name__ == "__main__":
    # Loading from dict/json
    config = Config.model_validate({
        "model": {
            "learning_rate": 1e-4,
            "batch_size": 64
        },
        "data": {
            "data_dir": "custom/path"
        }
    })

    # Or create with defaults
    config = Config()

    # Access nested attributes
    print(config.model.learning_rate)
    print(config.data.data_dir)