import pandas as pd
from transformers import AutoTokenizer
from src.tolkien_dataset import TolkienDataset


df = pd.read_json("tests/fixtures/test-tolkien-sentences.json")


def test_init():
    dataset = TolkienDataset(df, max_len=124)
    assert dataset.bos_token == "<|startoftext|>"
    assert dataset.eos_token == "<|endoftext|>"
    assert dataset.pad_token == "<|pad|>"
    # assert isinstance(dataset.tokenizer, AutoTokenizer)
    # assert dataset.data == ["a", "b", "c"]
    assert dataset.max_len == 124


def test_len():
    dataset = TolkienDataset(
        df,
        max_len=64,
    )
    assert len(dataset) == 50


def test_getitem():
    # doesn't test whether the text is encoded correctly
    dataset = TolkienDataset(df)
    assert (
        dataset[1]["sentence"] == "Seven for the Dwarf-lords in their halls of stone,\n"
    )


def test_encode_text():
    dataset = TolkienDataset(df, max_len=64)
    encodings_dict = dataset.encode_text("This is a different text!")

    assert len(encodings_dict["input_ids"]) == 64
    assert len(encodings_dict["attention_mask"]) == 64
