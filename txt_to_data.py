# turns the lotr_clean.txt into a json of sentences
import sys
import re
import json
import pandas as pd


def text_to_json(filename):
    sentences = get_sentences(filename)
    write_to_json(filename, sentences)


def get_sentences(filename):
    # splits text into "sentences" based on lines and punctuation
    sentences = []
    with open(filename, "r") as f:
        for line in f:
            line = line.lstrip()  # removes leading whitespace
            lines = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", line)
            [sentences.append(line) for line in lines if line != ""]

    print("First 20 sentences:")
    print(sentences[:20])

    return sentences


def write_to_json(filename, sentences):
    # writes sentences to csv
    jsonString = json.dumps(sentences)
    with open(filename[:-4] + "-sentences.json", "w") as f:
        f.write(jsonString)


if __name__ == "__main__":
    filename = sys.argv[1]
    text_to_json(filename)
