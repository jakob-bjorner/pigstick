import numpy as np

def reward_function(filter_function, reward_function, prompt, code):
    mask = [lint_filter(c) for c in code]
    rewards = reward_function(prompt[mask==True], code[mask==True])
    return rewards
