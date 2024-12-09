"""
Tokenizer class. 

Adapted from: 
https://github.com/Lightning-AI/lit-llama/blob/d513022842f7ee54c86595ce636d3133e35f8a8c/lit_llama/tokenizer.py
"""

import os
from pathlib import Path
from typing import Optional

import torch

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer

from config import TokenizerConfig


class Tokenizer:
    """
    Tokenizer for LLaMA.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        self.load(config)

    def load(self, config: TokenizerConfig) -> None:
        """
        Load the tokenizer from a model path. 
        """
        if config.type == "sentencepiece":
            print("Loading sentencepiece tokenizer...")
            self.processor = SentencePieceProcessor(model_file=str(config.tokenizer_path))
        elif config.type == "huggingface":
            print("Loading huggingface tokenizer...")
            self.processor = AutoTokenizer.from_pretrained(config.tokenizer_path)
        else:
            raise ValueError(f"Invalid tokenizer type: {config.type}")
        
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.pad_id = self.processor.pad_id()


    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(
        self,
        string: str,
        bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        tokens = self.processor.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.processor.decode(tokens.tolist())

   