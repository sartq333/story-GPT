# story-GPT
A simple GPT model trained from scratch on [tiny stories dataset](https://huggingface.co/roneneldan/TinyStories-33M/tree/main). It took around 5 hours to train the model for 50 epochs on a P100.

Model checkpoints can be found here: [model checkpoints](https://huggingface.co/Sartc/storyGPT/tree/main). Save these weights in a new folder and name that folder as checkpoints. Now run ```python3 inference.py``` for inferencing.

# Future Work/Things to do:

Upload model checkpoints. Done.

Implementation of a metric to evaluate the outputs (perplexity) and set a proper validation dataset.

Implement a proper directory structure for inference, so that model.generate() can be used easily. Done.

Replace the use of tiktoken in this code to make it completely independent by probably using [this](https://github.com/karpathy/minbpe).

A simple gradio UI/host on huggingface space (whichever is more convinent).

I've a couple of more things in my mind to improve this. Will be adding them in the future scope of work for this project if it is feasible to work on those (like dockerization and jit compilation to improve inference).

# References:

[Kaggle notebook by Tanay Mehta](https://www.kaggle.com/code/heyytanay/gpt-from-scratch-using-lightning-and-lance/notebook)
[Build LLMs from scratch book/github](https://github.com/rasbt/LLMs-from-scratch)

