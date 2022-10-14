from transformers import AutoTokenizer
from src.tolkien_dataset import TolkienDataset


def test_init():
    dataset = TolkienDataset(["a", "b", "c"], max_len=124)
    assert dataset.bos_token == "<|startoftext|>"
    assert dataset.eos_token == "<|endoftext|>"
    assert dataset.pad_token == "<|pad|>"
    # assert isinstance(dataset.tokenizer, AutoTokenizer)
    assert dataset.text_data == ["a", "b", "c"]
    assert len(dataset.text_data) == 3
    assert dataset.max_len == 124


def test_len():
    dataset = TolkienDataset(
        [
            "There once was a young man named Reese.",
            "The dragon attacked!",
            "Frodo went to bed.",
        ],
        max_len=64,
    )
    assert len(dataset) == 3


def test_getitem():
    # doesn't test whether the text is encoded correctly
    dataset = TolkienDataset(
        [
            "There once was a young man named Reese.",
            "The dragon attacked!",
            "Frodo went to bed.",
        ],
    )
    assert dataset[1]["text"] == "The dragon attacked!"


def test_encode_text():
    dataset = TolkienDataset(
        [
            "There once was a young man named Reese.",
            "The dragon attacked!",
            "Frodo went to bed.",
        ],
        max_len=64,
    )
    encodings_dict = dataset.encode_text("This is a different text!")

    assert len(encodings_dict["input_ids"]) == 64
    assert len(encodings_dict["attention_mask"]) == 64
