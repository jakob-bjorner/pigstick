"""Wrapper around different language models"""
import os 

import transformers
import torch

from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseModel:
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.llm = AutoModelForCausalLM.from_pretrained(filepath)
        self.tokenizer = AutoTokenizer.AutoTokenizer.from_pretrained(filepath)

    def update_model(self, filepath: str):
        self.llm = AutoModelForCausalLM.from_pretrained(filepath)
        self.tokenizer = AutoTokenizer.AutoTokenizer.from_pretrained(filepath)

    def generate(self, prompt: str, decoding_config: dict) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output_tokens = self.lm.generate(
            input_ids,
            decoding_config
        )
        # if decoder only model, then we need to isolate the
        # newly generated tokens as only those are watermarked, the input/prompt is not
        output_tokens = output_tokens[:, input_ids.shape[-1] :]
        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return output_text
