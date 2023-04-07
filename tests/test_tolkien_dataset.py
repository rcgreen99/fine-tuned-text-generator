import pandas as pd
from src.tolkien_dataset import TolkienDataset


df = pd.read_json("tests/fixtures/test-tolkien-sentences.json")
df = df.rename(columns={0: "sentences"})


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
    dataset = TolkienDataset(df)
    assert (
        dataset[1]["sentence"] == "Seven for the Dwarf-lords in their halls of stone,\n"
    )
    assert len(dataset[1]["input_ids"]) == 256
    assert len(dataset[1]["attention_mask"]) == 256
    assert dataset.tokenizer.decode(dataset[1]["input_ids"][:15]) == (
        "<|startoftext|>Seven for the Dwarf-lords in their halls of stone,\n<|endoftext|>"
    )


def test_encode_text():
    dataset = TolkienDataset(df, max_len=64)
    encodings_dict = dataset.encode_text("This is a different text!")

    assert len(encodings_dict["input_ids"]) == 64
    assert len(encodings_dict["attention_mask"]) == 64
