import torch

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, IntervalStrategy

# from transformers import Traniner

from src.tolkien_dataset_builder import TolkienDatasetBuilder
from src.training.training_session_arg_parser import TrainingSessionArgParser
from src.training.trainer import Trainer


class TrainingSession:
    def __init__(self, args):
        self.args = args

        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-125M",
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
        )

    def run(self):
        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.create_trainer()
        self.trainer.fit()

    def create_datasets(self):
        builder = TolkienDatasetBuilder(self.args.filename)
        self.train_dataset, self.val_dataset = builder.build_datasets()

    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size
        )

    def create_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-125M"
        ).cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))

    # def create_trainer_auto(self):
    #     training_args = TrainingArguments(
    #         output_dir="./results",
    #         num_train_epochs=5,
    #         logging_steps=5000,
    #         save_strategy=IntervalStrategy.NO,
    #         per_device_train_batch_size=2,
    #         per_device_eval_batch_size=2,
    #         warmup_steps=100,
    #         weight_decay=0.01,
    #         logging_dir="./logs",
    #     )
    #     Trainer(
    #         model=self.model,
    #         args=training_args,
    #         train_dataset=self.train_dataset,
    #         eval_dataset=self.val_dataset,
    #         data_collator=lambda data: {
    #             "input_ids": torch.stack([f["input_ids"] for f in data]),
    #             "attention_mask": torch.stack([f["attention_mask"] for f in data]),
    #             "labels": torch.stack([f["input_ids"] for f in data]),
    #         },
    #     ).train()

    #     self.model.save_pretrained("./results")

    #     generated = self.tokenizer(
    #         "<|startoftext|>", return_tensors="pt"
    #     ).input_ids.cuda()

    #     sample_outputs = self.model.generate(
    #         generated,
    #         do_sample=True,
    #         top_k=50,
    #         bos_token_id=self.tokenizer.bos_token_id,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         max_length=300,
    #         top_p=0.95,
    #         temperature=1.9,
    #         num_return_sequences=20,
    #     )
    #     for i, sample_output in enumerate(sample_outputs):
    #         print(
    #             "{}: {}".format(
    #                 i, self.tokenizer.decode(sample_output, skip_special_tokens=True)
    #             )
    #         )

    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            learning_rate=self.args.learning_rate,
            epochs=self.args.epochs,
        )


if __name__ == "__main__":
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.run()
