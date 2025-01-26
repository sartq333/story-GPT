# story-GPT
A simple GPT model trained from scratch on [tiny stories dataset](https://huggingface.co/roneneldan/TinyStories-33M/tree/main). It took around 5 hours to train the model for 50 epochs on a P100.

Model checkpoints can be found here: [model checkpoints](https://huggingface.co/Sartc/storyGPT/tree/main). Save these weights in a new folder and name that folder as checkpoints. Now run ```python3 inference.py``` for inferencing.

SS of model's response (via terminal on CPU):

![image](https://github.com/user-attachments/assets/feb3f6fe-2813-4150-802b-8295ff814a61)

SS of model's response (via gradio UI on CPU):
![image](https://github.com/user-attachments/assets/95dde6bd-88f5-4991-a517-d651e6c208ef)



# Future Work/Things to do:

~Upload model checkpoints. Done.~

Implementation of a metric to evaluate the outputs (perplexity) and set a proper validation dataset.

~Implement a proper directory structure for inference, so that model.generate() can be used easily. Done.~

Replace the use of tiktoken in this code to make it completely independent by probably using these as references https://github.com/karpathy/minbpe, https://sebastianraschka.com/blog/2025/bpe-from-scratch.html.

~A simple gradio UI/host on huggingface space (whichever is more convinent).~

I've a couple of more things in my mind to improve this. Will be adding them in the future scope of work for this project if it is feasible to work on those (like dockerization and jit compilation to improve inference).

# References:

[Kaggle notebook by Tanay Mehta](https://www.kaggle.com/code/heyytanay/gpt-from-scratch-using-lightning-and-lance/notebook)

[Build LLMs from scratch book/github](https://github.com/rasbt/LLMs-from-scratch)
