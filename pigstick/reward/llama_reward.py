import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn

def llama_reward_hidden(input_ids, attention_mask, labels, model_filepath):
    """
    Compute the loss of a CodeLlama-like model given input IDs, attention mask, and labels.

    Args:
    - model_name (str): The name of the pre-trained CodeLlama model (or similar model).
    - input_ids (torch.Tensor): The input token IDs (shape: [batch_size, sequence_length]).
    - attention_mask (torch.Tensor): The attention mask (shape: [batch_size, sequence_length]).
    - labels (torch.Tensor): The ground truth labels (shape: [batch_size, sequence_length]).

    Returns:
    - loss (torch.Tensor): The computed loss.
    """

    # Load the pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_filepath)

    # Ensure the model is in evaluation mode (disables dropout layers etc.)
    model.eval()

    # Move tensors to the device of the model (e.g., CUDA if available)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # Forward pass through the model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # The loss is stored in the 'loss' attribute of the output object
    loss = outputs.loss

    return loss
