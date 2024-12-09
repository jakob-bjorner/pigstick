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
from typing import Optional

import numpy as np
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import generate, generate_prompt
from dataloader import load_datasets, DataLoader 

@dataclass
class TrainingConfig:
    # Training loop settings
    instruction_tuning: bool = True
    eval_interval: int = 1000
    save_interval: int = 1000
    eval_iters: int = 100
    log_interval: int = 100
    devices: int = 1
    
    # Model hyperparameters
    learning_rate: float = 3e-5
    batch_size: int = 128
    micro_batch_size: int = 8
    epoch_size: int = 30000
    num_epochs: int = 3
    weight_decay: float = 0.0
    block_size: int = 512
    warmup_iters: int = 100
    # Derived properties
    @property
    def gradient_accumulation_iters(self) -> int:
        return self.batch_size // self.micro_batch_size
    
    @property
    def max_iters(self) -> int:
        return self.num_epochs * (self.epoch_size // self.micro_batch_size) // self.devices
    
    def __post_init__(self):
        assert self.gradient_accumulation_iters > 0, "Batch size must be larger than micro batch size"


def main(
    data_dir: str = "data/alpaca",
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = "out/full/alpaca",
    config: Optional[TrainingConfig] = None,
):
    """ 
    
    """
    if config is None:
        config = TrainingConfig()
    
    fabric = L.Fabric(accelerator="cuda", 
                      devices=config.devices, 
                      precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(1337)

    os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config.block_size = config.block_size

    checkpoint = torch.load(pretrained_path)  

    with fabric.device:
        torch.set_default_tensor_type(torch.HalfTensor)
        model = LLaMA(config).bfloat16()
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False) 

    model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, foreach=False)
    optimizer = fabric.setup_optimizers(optimizer)

    train(fabric, model, optimizer, train_data, val_data, out_dir, config)

    # Save the final checkpoint at the end of training
    save_model_checkpoint(fabric, model, os.path.join(out_dir, "lit-llama-full-finetuned.pth"))


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
    config: TrainingConfig,
) -> None:
    """
    The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    model.train()

    for iter_num in range(config.max_iters):

        is_accumulating = (iter_num + 1) % config.gradient_accumulation_iters != 0

        if step_count <= config.warmup_iters:
            # linear warmup
            lr = config.learning_rate * step_count / config.warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()
        
        input_ids, targets = get_batch(fabric, train_data)
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / config.gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % config.eval_interval == 0:
                val_loss = validate(fabric, model, val_data, config)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")

            if step_count % config.save_interval == 0:
                print(f"Saving weights to {out_dir}")
                save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))

        dt = time.time() - t0
        if iter_num % config.log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
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
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, config: TrainingConfig) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(config.eval_iters)
    for k in range(config.eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def get_batch(fabric: L.Fabric, data: list):
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
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description='LLaMA SFT Training')
    parser.add_argument('--data_dir', type=str, default="data/alpaca")
    parser.add_argument('--pretrained_path', type=str, default="checkpoints/lit-llama/7B/lit-llama.pth")
    parser.add_argument('--out_dir', type=str, default="out/full/alpaca")
    
    args = parser.parse_args()
    config = TrainingConfig()  # You can add CLI args to override config values if needed
    main(data_dir=args.data_dir, pretrained_path=args.pretrained_path, out_dir=args.out_dir, config=config)