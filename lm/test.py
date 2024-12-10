from pigstick.lm.base_lm import LM 
from transformers import GenerationConfig

def main():
    
    # define language model
    lm_path = "/srv/share5/nghia6/codebases/meta-llama/Llama-2-7b-chat-hf" 
    lm = LM(lm_path=lm_path)

    # define decoding strategies TODO
    decoding_config = GenerationConfig.from_pretrained(
        lm_path,
        do_sample=False,         # Enables sampling
        max_length=50,          # Maximum length of the generated sequence
        top_k=50,               # Optional: Use top-k sampling
        top_p=0.9,              # Optional: Use nucleus sampling
        temperature=1.0,        # Adjust randomness; higher values = more randomness
    )

    # test output
    prompt = "Tell me a scary story: "
    output = lm.generate(prompt=prompt, decoding_config=decoding_config)
    print(f"Prompt={prompt}")
    print(f"Output={output}")

if __name__ == "__main__":
    main()