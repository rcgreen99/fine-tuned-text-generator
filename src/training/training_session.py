from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from src.tolkien_dataset_builder import TolkienDatasetBuilder
from src.training.training_session_arg_parser import TrainingSessionArgParser


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.create_trainer()
        self.trainer.fit()

    def create_datasets(self):
        builder = TolkienDatasetBuilder(self.args.filename)
        self.train_dataset, self.val_dataset = builder.build()

    def create_dataloaders(self):
        pass

    def create_model(self):
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    def create_trainer(self):
        pass


if __name__ == "__main__":
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.run()
