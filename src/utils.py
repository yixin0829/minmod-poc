import unicodedata

import pandas as pd
import tiktoken
from pydantic.v1 import BaseModel


def count_file_tokens(file_path: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in the input text file.
    """
    if not file_path:
        raise ValueError("file_path is empty.")

    with open(file_path, "r") as f:
        text = f.read()
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
