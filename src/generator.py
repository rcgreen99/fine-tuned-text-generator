import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


class Generator:
    def __init__(self, model_name, model_dir):
        self.model_name = model_name
        self.model_dir = model_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.model.eval()

    def generate(self, prompt):
        prompt = "<|startoftext|>" + prompt

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        sample_outputs = self.model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            max_length=300,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=3,
        )
        for i, sample_output in enumerate(sample_outputs):
            print(
                "{}: {}".format(
                    i, self.tokenizer.decode(sample_output, skip_special_tokens=True)
                )
            )

    def generate_loop(self):
        while True:
            prompt = input("Enter prompt: ")
            self.generate(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()

    generator = Generator(args.model_name, args.model_dir)
    generator.generate_loop()
