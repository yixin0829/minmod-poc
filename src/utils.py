import unicodedata

import pandas as pd
import tiktoken


def count_file_tokens(file_path: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in the input text file.
    """
    if file_path:
        with open(file_path, "r") as f:
            text = f.read()
    else:
        raise ValueError("file_path is empty.")
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def count_str_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in the input text.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def normalize_unicode(text: str) -> str:
    """
    Normalize the input text to NFKC form.
    """
    return unicodedata.normalize("NFC", text)


def convert_to_numeric(x: str):
    try:
        return pd.to_numeric(x)
    except ValueError:
        return x


# Max context window of GPT-4-turbo is 128,000 tokens.
# print(count_file_tokens("data/asset/parsed_result/Bleiberg_Pb_Zn_5-2017/result.txt"))
# print(
#     count_file_tokens("data/asset/parsed_result/Bongará_Zn_3-2019/result.txt")
# )  # 73370 tokens ~= 200 pages


# print(r"INFERRED MINERAL RESOURCES, BONGAR\u00c1 MINE PROJECT as of January, 2019 \u2013 10% Zn Cut-Off ")
# normalized_str = normalize_unicode("INFERRED MINERAL RESOURCES, BONGAR\u00c1 MINE PROJECT as of January, 2019 \u2013 10% Zn Cut-Off ")
# print(normalized_str)
