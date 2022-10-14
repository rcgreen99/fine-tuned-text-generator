import pandas as pd
from src.tolkien_dataset_builder import TolkienDatasetBuilder
from src.tolkien_dataset import TolkienDataset

filename = "tests/fixtures/test-tolkien-sentences.json"


def test_init():
    builder = TolkienDatasetBuilder(filename)
    assert builder.filename == filename
    assert builder.val_percent == 0.1


def test_build_datasets():
    builder = TolkienDatasetBuilder(filename, 0.2)
    train_dataset, val_dataset = builder.build_datasets()
    assert train_dataset is not None
    assert val_dataset is not None
    assert len(train_dataset) * 1.25 == len(val_dataset) * 5


def test_random_split_dataset():
    builder = TolkienDatasetBuilder(filename, 0.2)
    df = pd.read_json(filename)
    dataset = TolkienDataset(df)
    train_dataset, val_dataset = builder.random_split_dataset(dataset)
    assert train_dataset is not None
    assert val_dataset is not None
    assert train_dataset[0] is not None
