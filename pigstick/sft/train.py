"""
Reference implementation for regular SFT. Taken from: 

https://github.com/Lightning-AI/lit-llama/blob/d513022842f7ee54c86595ce636d3133e35f8a8c/finetune/full.py
"""

import sys
from pathlib import Path
import os
import time
from functools import partial
from dataclasses import dataclass
from typing import Optional, List, Any, Union

import random
import numpy as np
import torch

from config import Config, load_config, ModelConfig, DataConfig, TrainingLoopConfig
from model import Model, generate, generate_prompt
from tokenizer import Tokenizer
from dataloader import load_datasets, DataLoader 


def logging(msg: str) -> None:
    """ 
    Log functions for training. For now, just print to console. 
    """
    print(msg)


def main(config_path: str) -> None:
    """ 
    Basic training loop for SFT. 
    """
    config = load_config(config_path)
    
    # set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed) 

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.out_dir, exist_ok=True)

    # initialize training 
    train_data, val_data = load_datasets(data_dir=config.data_dir)
    
    # load model
    if config.pretrained_path:
        checkpoint = torch.load(config.pretrained_path)
        model = Model(config.model).to(device).bfloat16()
        model.load_state_dict(checkpoint, strict=False)
    else:
        model = Model(config.model).to(device).bfloat16()

    # load tokenizer
    tokenizer = Tokenizer(config.tokenizer_path)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # train
    train(model, optimizer, train_data, val_data, config.train)

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(config.out_dir, "model-finetuned.pth"))


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    config: TrainingLoopConfig,
) -> None:
    """
    The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    model.train()
    device = next(model.parameters()).device

    for iter_num in range(config.max_iters):

        if step_count <= config.warmup_iters:
            # linear warmup
            lr = config.learning_rate * step_count / config.warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()
        
        input_ids, targets = get_batch(train_data, config, device)
        
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        loss = loss / config.gradient_accumulation_iters
        
        loss.backward()

        if (iter_num + 1) % config.gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % config.eval_interval == 0:
                val_loss = validate(model, val_data, config)
                print(f"step {iter_num}: val loss {val_loss:.4f}")

            if step_count % config.save_interval == 0:
                print(f"Saving weights to {config.out_dir}")
                torch.save(model.state_dict(), 
                          os.path.join(config.out_dir, f"iter-{iter_num:06d}-ckpt.pth"))

        dt = time.time() - t0
        if iter_num % config.log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")



def generate_response(model, tokenizer, instruction, config):
    """
    Generate a response from the model. 
    """
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if config.instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=config.block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(model: torch.nn.Module,
             tokenizer: Tokenizer,
             val_data: np.ndarray, 
             config: TrainingLoopConfig) -> float:
    """
    Validate the model. 
    """
    print("Validating ...")
    model.eval()
    device = next(model.parameters()).device
    losses = torch.zeros(config.eval_iters)
    
    for k in range(config.eval_iters):
        input_ids, targets = get_batch(val_data, config, device)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, tokenizer, instruction, config)
    print(instruction)
    print(output)

    model.train()
    return out.item()


def loss_fn(logits, targets):
    """
    Compute the loss for the model. 
    """
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def get_batch(data: list, config: TrainingLoopConfig, device: torch.device):
    """
    Get a batch of data from the dataset. 
    """
    ix = torch.randint(len(data), (config.micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    return x.to(device), y.to(device)


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description='LLaMA SFT Training')
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()

    config = load_config(args.config)  
    main(config=config)