import torch
import torch.nn as nn
import gradio as gr
from transformers import GPT2TokenizerFast
from model import Config, GPT

config = Config()
gpt = GPT.load_from_checkpoint("checkpoints/epoch=49-step=43600.ckpt")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def generate(model, prompt, max_tokens, temperature=0.7):
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            prompt = prompt[:, :config.context_len]
            logits = model(prompt)
            logits = logits[:, -1, :] / temperature 
            logit_probs = nn.functional.softmax(logits, dim=-1)
            next_prompt = torch.multinomial(logit_probs, num_samples=1)
            prompt = torch.cat((prompt, next_prompt), dim=1)
    
    return prompt

def generate_text(prompt):
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt')
    generated_tokens = generate(gpt, prompt_tokens, max_tokens=config.context_len, temperature=0.7)
    generated_text = tokenizer.decode(generated_tokens.tolist()[0])
    return generated_text

def create_interface():
    iface = gr.Interface(
        fn=generate_text,
        inputs=gr.Textbox(label="Enter your prompt"),
        outputs=gr.Textbox(label="Generated Text"),
        title="GPT Text Generator",
        description="Generate text using the trained GPT model"
    )
    return iface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()