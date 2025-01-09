import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast
from model import Config, GPT

config = Config()
gpt = GPT.load_from_checkpoint("checkpoints/epoch=49-step=43600.ckpt")

def generate(model, prompt, max_tokens, temperature=0.7):
    model.eval()
    for _ in range(max_tokens):
        prompt = prompt[:, :config.context_len]
        logits = model(prompt)
        logits = logits[:, -1, :] / temperature
        logit_probs = nn.functional.softmax(logits, dim=-1)
        next_prompt = torch.multinomial(logit_probs, num_samples=1)
        prompt = torch.cat((prompt, next_prompt), dim=1)
    return prompt

def run(prompt):
    if prompt=="bye":
        print("Bye!")
        return 
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # gpt = gpt.to('cuda')
    # prompt = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    prompt = tokenizer.encode(prompt, return_tensors='pt')
    generated_text = generate(gpt, prompt, max_tokens=config.context_len, temperature=0.7)
    generated_text = tokenizer.decode(generated_text.tolist()[0])
    print(generated_text)
    prompt = input("Your prompt:")
    run(prompt)

if __name__ == "__main__":
    prompt = input("Your prompt:")
    run(prompt)
