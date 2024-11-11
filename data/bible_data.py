from typing import Callable
from data import data_utils
import numpy as np


def read_bible_data(file_path) -> tuple[str, list[str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        text_lines = f.readlines()
    bible_lines = text_lines[2:]

    sections = []
    current_chapter_name = ""
    section_lines = []

    for line_index in range(len(bible_lines) - 1):
        current_line = bible_lines[line_index]

        chapter_name = current_line.split()[0]
        if chapter_name != current_chapter_name:
            if len(section_lines) > 0:
                sections.append("\n".join(section_lines))
            section_lines.clear()
            current_chapter_name = chapter_name
        else:
            section_lines.append(current_line)

    if len(section_lines) > 0:
        sections.append("\n".join(section_lines))

    bible_text = "\n".join(text_lines[2:])  # exclude first two lines with metadata

    return bible_text, sections


def prepare_bible_data(sections: list[str], tokenization_func: Callable[[str], list[int]], eot_token: int, train_output_path, val_output_path):
    tokens = []
    max_validation_tokens = 32768
    for section in sections:
        tokens.append(eot_token)
        tokens.extend(tokenization_func(section))

    tokens_array = np.asarray(tokens)
    data_utils.write_datafile(val_output_path, tokens_array[:max_validation_tokens], "gpt-2")
    data_utils.write_datafile(train_output_path, tokens_array[max_validation_tokens:], "gpt-2")


def convert_bible_data_for_gpt(filepath: str, train_output_path: str, val_output_path: str):
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc._special_tokens["<|endoftext|>"]  # end of text token
    tokenize = lambda s: enc.encode_ordinary(s)
    _, sections = read_bible_data(filepath)
    prepare_bible_data(sections, tokenize, eot_token, train_output_path, val_output_path)


if __name__ == "__main__":
    filepath = "bible.txt"
    bible_txt, sections = read_bible_data(filepath)
    print("Bible length:", len(bible_txt))

    convert_bible_data_for_gpt(filepath, "train_bible.bin", "val_bible.bin")
    print("Done")
