import os
import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, IntervalStrategy, Trainer

# from transformers import Traniner

from src.tolkien_dataset_builder import TolkienDatasetBuilder
from src.training.training_session_arg_parser import TrainingSessionArgParser

# from src.training.trainer import Trainer

os.environ["WANDB_DISABLED"] = "true"

# MODEL_NAME = "EleutherAI/gpt-neo-125M"
MODEL_NAME = "facebook/opt-350m"


class TrainingSession:
    def __init__(self, args):
        self.args = args

        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
        )

    def run(self):
        self.create_datasets()
        # self.create_dataloaders()
        self.create_model()
        self.create_trainer()
        self.trainer.train()

    def create_datasets(self):
        builder = TolkienDatasetBuilder(self.args.filename, MODEL_NAME)
        self.train_dataset, self.val_dataset = builder.build_datasets()

    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size
        )

    def create_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))

    def create_trainer(self):
        # self.trainer = Trainer(
        #     model=self.model,
        #     train_dataloader=self.train_dataloader,
        #     val_dataloader=self.val_dataloader,
        #     learning_rate=self.args.learning_rate,
        #     epochs=self.args.epochs,
        # )
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        training_args = TrainingArguments(
            output_dir=f"./results/{time}",
            num_train_epochs=1,
            logging_steps=5000,
            # save_strategy=IntervalStrategy.NO,
            per_device_train_batch_size=16,
            # per_device_eval_batch_size=8,
            # warmup_steps=100,
            # weight_decay=0.01,
            logging_dir="./logs",
        )
        self.trainer = Trainer(
            model=self.model,
            # args=self.args,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=lambda data: {
                "input_ids": torch.stack([f["input_ids"] for f in data]),
                "attention_mask": torch.stack([f["attention_mask"] for f in data]),
                "labels": torch.stack([f["input_ids"] for f in data]),
            },
        )


if __name__ == "__main__":
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.run()
