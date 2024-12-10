# from pigstick.lm.base_lm import LM 
# from lm.base_lm import LM 
# from pigstick.eval import evaluateModel
from transformers import GenerationConfig
from datasets import load_dataset

import os 

import transformers
import torch

from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseModel:
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.llm = AutoModelForCausalLM.from_pretrained(filepath)
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)

    def update_model(self, filepath: str):
        self.llm = AutoModelForCausalLM.from_pretrained(filepath)
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)

    def generate(self, prompt: str, decoding_config: dict) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output_tokens = self.llm.generate(
            input_ids,
            decoding_config
        )
        # if decoder only model, then we need to isolate the
        # newly generated tokens as only those are watermarked, the input/prompt is not
        output_tokens = output_tokens[:, input_ids.shape[-1] :]
        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return output_text

def format_python_output(output):

    # format generated code 
    # generated_code = output["prompt"] + output["code"]
    generated_code = output["prompt"] + output["gold_code"]
    print(f"generated_code={generated_code}")

    # format unit test TODO 
    print(f"unit test={output['unit test values']}")

    # concatenate

def evaluate(outputs):
    
    exp_dir = "/srv/share5/nghia6/codebases/pigstick/outputs/test"
    os.makedirs(exp_dir, exist_ok=True)

    for i, output in enumerate(outputs): 

        # construct python string TODO
        format_python_output(output)

        # output to path TODO 

        # run python subprocess TODO 
    
        # pass into code at k TODO

def main():
    
    # define language model
    lm_path = "/srv/share5/nghia6/codebases/codellama/CodeLlama-7b-hf" 
    lm = BaseModel(filepath=lm_path)

    # define decoding strategies TODO
    decoding_config = GenerationConfig.from_pretrained(
        lm_path,
        do_sample=True,         # Enables sampling
        max_length=50,          # Maximum length of the generated sequence
        top_k=50,               # Optional: Use top-k sampling
        top_p=0.9,              # Optional: Use nucleus sampling
        temperature=1.0,        # Adjust randomness; higher values = more randomness
        num_return_sequences=3  # Generate multiple sequences
    )

    # loading data TODO 
    dataset = load_dataset("openai_humaneval")["test"]
    max_examples = 1
    test_dataset = []
    print(dataset.select(range(max_examples))[0].keys())
    for i, example in enumerate(dataset.select(range(max_examples))):
        # Extract the instructions
        task_name = example['task_id']
        prompt = example['prompt']
        # print(f"\n=== Task {task_name} ===")
        # print(f"Instruction:\n{prompt}")

        test_dataset.append({
            "prompt": prompt,
            "code": example['canonical_solution'],
            "unit test values": example['test'],
        })

    # generate outputs
    outputs = [] 
    for example in test_dataset: 
        output = lm.generate(prompt=example["prompt"], decoding_config=decoding_config)
        outputs.append({
            "prompt": example["prompt"], 
            "gold_code": example["code"],
            "generated_code": output,
            "unit test values": example['unit test values'], 
        })
    print(outputs[0])

    # evaluate generation TODO
    evaluate(outputs)


if __name__ == "__main__":
    main()