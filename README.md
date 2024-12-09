# pigstick
hackathon for 2024 kargo lab

[Project Doc](https://docs.google.com/document/d/1Mm3LFp2ljiG0whlcMjvvCJ-S9P8eA71jAPAbhOQVYuo/edit?usp=sharing)


In the root directory of this repo run the following command with your python 3.10.* environment of choice.
(I suggest making a specific environment for this project using miniconda)
``` bash
python3 -m pip install -e .
```

## Progress Thus Far

There were 3 teams working on various aspects of pigstick - Model & Eval Team, SFT Team, and Reward Model Team

### Model & Eval Team

### SFT Team 

All the work for the SFT loop is in the `pigstick/sft` directory. 

`train.py`: main training loop. 
`model.py`: placeholder model and tokenizer. 
`tokenizer.py`: tokenizer for the model. 
`config.py`: config for the SFT loop. 

`.old/`: old code, notes, etc.

`data.ipynb`: sandbox for loading datasets  
`dataloader.py`: data loading and preprocessing. 

Most of the SFT training code was adapted from the [Lightning-AI lit-llama implementation](https://github.com/Lightning-AI/lit-llama/tree/d513022842f7ee54c86595ce636d3133e35f8a8c/finetune). 


### Pending Tasks

The main work left to do is to finish implementing the dataloader: 
    - Support special tokens: prompt, code, etc. See [StarCoder](https://huggingface.co/bigcode/starcoder) and [CodeLlama](https://huggingface.co/codellama/CodeLlama-7b-hf) papers for this?  
    - scheduled training: feed in training data in a specific order: 
        - Ultrafeed 
        - Github Code
        - BigBench Code 



### Reward Model Team


