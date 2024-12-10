# from pigstick.lm.base_lm import LM 
# from lm.base_lm import LM 
# from pigstick.eval import evaluateModel
from transformers import GenerationConfig
from datasets import load_dataset

import os 

import transformers
import torch
import subprocess

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
    # print(f"generated_code={generated_code}")

    # format unit test TODO 
    # print(f"unit test={output['unit test values']}")

    # concatenate
    python_output = generated_code + output['unit test values'] + f"check({output['function_name']})" 
    print(f"python output={python_output}")
    return python_output

def evaluate(outputs):
    
    exp_dir = "/srv/share5/nghia6/codebases/pigstick/outputs/test"
    os.makedirs(exp_dir, exist_ok=True)

    for i, output in enumerate(outputs): 

        # construct python string 
        python_output = format_python_output(output)

        # output to path 
        python_filepath = os.path.join(exp_dir, f"file_{i}.py") 
        with open(python_filepath, "w", encoding="utf-8") as f:
            f.write(python_output)

        # run python subprocess TODO
        result = subprocess.run(
            ["python", python_filepath],
            capture_output=True,
            text=True
        )
        print("Output:")
        print(result.stdout)
        if result.stderr: # failure case 
            print("Errors:")
            print(result.stderr) 

        else: # correct case
            pass
    
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

    # generate outputs
    outputs = [] 
    for i, example in enumerate(dataset.select(range(max_examples))):
        output = lm.generate(prompt=example["prompt"], decoding_config=decoding_config)
        outputs.append({
            "prompt": example["prompt"], 
            "gold_code": example["canonical_solution"],
            "generated_code": output,
            "unit test values": example['test'], 
            "function_name": example["entry_point"]
        })
    print(outputs[0])

    # evaluate generation TODO
    evaluate(outputs)


if __name__ == "__main__":
    main()