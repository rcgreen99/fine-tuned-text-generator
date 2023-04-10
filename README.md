# tolkien-text-generator
Fine tuning HuggingFace CausalLMs to generate text resembling J. R. R. Tolkien's writing.

## About
This repo contains the code to fine tune a causalLM of your choice as well as easily generating your own text.

Personally, I tested this using the the models `EleutherAI/gpt-neo-125M` and `facebook/opt-350M`. In theory, any other hugging face CausalLM can be used, but given my memeory constrains (32GB RAM w/ GTX 3090) I was limited to relatively smaller models. For the most part, larger and more recently released models should produce better results.

While I only fine-tuned on Tolkien's writings, any text can be used. For the best results, I suggest splitting text into roughly paragraph length. 

## Usage

**To fine-tune a model:**

    bin/train.sh --model_name model_name
    
model_name defaults to `facebook/opt-350M`. Run `bin/train.sh -h` for a list of other parameters.

**To generate text:**

    bin/gen.sh --model_name model_name --model_dir model_dir

Where `model_name` is the hugging face pre trained model, and `model_dir` is the fine tuned model.

`model_name` defaults to facebook/opt-350M
 