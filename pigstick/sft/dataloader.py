""" 
Dataloader for SFT pipeline. 

"""
from typing import Tuple

import numpy as np



def load_datasets(data_dir: str) -> Tuple[np.ndarray, np.ndarray]: 
    """ 
    Load datasets. 
    """
    pass



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



