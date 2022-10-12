from transformers import AutoTokenizer
from torch.utils.data import Dataset


class TolkienDataset(Dataset):
    def __init__(self, text_list, max_len=768):
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-125M",
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
        )

        self.text_data = text_list
        self.max_len = max_len

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        encodings_dict = self.encode_text(text)

        # no labels because input_ids will be used as "labels" for CausalLM
        return {
            "text": text,
            "input_ids": encodings_dict["input_ids"],
            "attention_mask": encodings_dict["attention_mask"],
        }

    def encode_text(self, text):
        encodings_dict = self.tokenizer(
            self.bos_token + text + self.eos_token,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        return encodings_dict
