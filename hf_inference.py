import os.path
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import GPT2TokenizerFast
from model import Config, GPT  
import torch.nn as nn

config = Config()

def load_safetensors(path):
    state_dict = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict

def load_local(path):
    return load_safetensors(path)
    
def load_from_hf(repo_id):
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename="storyGPT.safetensors"  
    )
    return load_safetensors(file_path)

def load_model(repo_id, local_file):
    if repo_id:
        state_dict = load_from_hf(repo_id)
    elif local_file:
        state_dict = load_local(local_file)
    else:
        raise ValueError("Must provide either repo_id or local_file")
            
    model = GPT(config)   
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate(model, prompt, max_tokens, temperature=0.7):
    for _ in range(max_tokens):
        prompt = prompt[:, :config.context_len]
        logits = model(prompt)
        logits = logits[:, -1, :] / temperature
        logit_probs = nn.functional.softmax(logits, dim=-1)
        next_prompt = torch.multinomial(logit_probs, num_samples=1)
        prompt = torch.cat((prompt, next_prompt), dim=1)
    return prompt

def run(prompt):
    if prompt.lower() == "bye":
        print("Bye!")
        return
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():  # Disable gradient calculation
        generated = generate(gpt, inputs, 
                           max_tokens=config.context_len,
                           temperature=0.7)
    
    print(tokenizer.decode(generated[0].cpu().numpy()))
    new_prompt = input("Your prompt: ")
    run(new_prompt)

if __name__ == "__main__":
    
    file_path="storyGPT.safetensors"

    if os.path.exists(file_path):
        gpt = load_model(False, file_path)
    else:
        gpt = load_model("sartc/storyGPT", False)

    prompt = input("Your prompt: ")
    run(prompt)

