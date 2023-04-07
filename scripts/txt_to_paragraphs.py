import sys
import re
import json
import pandas as pd


def text_to_json(filename):
    sentences = get_paragraphs(filename)
    write_to_json(filename, sentences)


# splits text into paragraphs based on each line
def get_paragraphs(filename):
    paragraphs = []
    with open(filename, "r") as f:
        for line in f:
            line = line.lstrip()
            if line != "":
                paragraphs.append(line)
    return paragraphs


def write_to_json(filename, sentences):
    # writes sentences to csv
    jsonString = json.dumps(sentences)
    with open(filename[:-4] + "-paragraphs.json", "w") as f:
        f.write(jsonString)


if __name__ == "__main__":
    filename = sys.argv[1]
    text_to_json(filename)
