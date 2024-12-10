""" 
Dataloader for SFT pipeline. 

"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from datasets import load_dataset

def load_datasets(
    data_dir: str, 
    datasets: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]: 
    """ 
    Load datasets. 
    """
    # Available datasets and their loading methods
    dataset_loaders = {
        'ultrainteract': lambda: _load_ultrainteract(),
        'github': lambda: _load_github(),
        'bcb': lambda: _load_bcb()
    }
    
    # Determine which datasets to load
    if datasets is None:
        datasets = list(dataset_loaders.keys())
    
    # Validate input
    invalid_datasets = set(datasets) - set(dataset_loaders.keys())
    if invalid_datasets:
        raise ValueError(f"Invalid dataset names: {invalid_datasets}. "
                         f"Valid options are: {list(dataset_loaders.keys())}")
    
    # Load specified datasets
    loaded_dataframes = []
    for dataset_name in datasets:
        df = dataset_loaders[dataset_name]()
        loaded_dataframes.append(df)
    
    df_combined = pd.concat(loaded_dataframes, ignore_index=True)
    
    # Convert to numpy arrays
    texts = df_combined['content'].values
    
    return texts, labels

def _load_ultrainteract() -> pd.DataFrame:
    """Load UltraInteract dataset"""
    df = pd.read_parquet(
        "https://huggingface.co/datasets/openbmb/UltraInteract_sft/resolve/main/0000_sft.parquet"
    )
    df['content'] = df['instruction'] + ' ' + df['response']
    return df[['content']]

def _load_github() -> pd.DataFrame:
    """Load GitHub Python code dataset"""
    ds_github = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Python"])
    ds_github_20p = ds_github.shuffle(seed=42).take(30000)
    df_github = pd.DataFrame(ds_github_20p)
    df_github = df_github[df_github["language"] == "Python"]
    df_github = df_github.rename(columns={"code": "content"})
    return df_github[['content']]

def _load_bcb() -> pd.DataFrame:
    """Load BigCodeBench dataset"""
    df = pd.read_parquet(
        "https://huggingface.co/datasets/bigcode/bigcodebench/resolve/main/data/v0.1.0-00000-of-00001.parquet"
    )
    df['content'] = df['instruct_prompt'] + ' ' + df['canonical_solution']
    return df[['content']]



class DataLoader:
    """
    DataLoader for Supervised Fine-Tuning (SFT) pipeline.
    
    Attributes:
        data_dir (str): Directory containing the training data
        batch_size (int): Size of each training batch
        shuffle (bool): Whether to shuffle data between epochs
        tokenizer: Tokenizer instance for text processing
        max_length (int): Maximum sequence length for inputs
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        shuffle: bool = True,
        tokenizer = None,
        max_length: int = 512
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = None
        
    def load_data(self):
        """Load and preprocess the training data from data_dir"""
        pass
        
    def tokenize_data(self, text: str):
        """Tokenize input text using the specified tokenizer"""
        pass
    
    def create_prompt_completion_pair(self, sample):
        """Create formatted prompt-completion pairs for training"""
        pass
        
    def __len__(self):
        """Return the total number of samples"""
        pass
        
    def __getitem__(self, idx):
        """Get a single training sample"""
        pass
        
    def get_batch(self):
        """Generate batches of training data"""
        pass
        
    def shuffle_data(self):
        """Shuffle the training data when self.shuffle is True"""
        pass



