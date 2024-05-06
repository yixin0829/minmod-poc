import os
import unicodedata

import numpy as np
import pandas as pd
import tiktoken
from loguru import logger
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


def load_doc(doc_path: str) -> tuple[str, str]:
    """
    Load the document from the given file path.

    Args:
        doc_path (str): The path to the document file.

    Returns:
        tuple[str, str]: A tuple containing the file name and the content of the document.
    """
    logger.info(f"Loading document from {doc_path}")
    with open(doc_path, "r") as f:
        doc = f.read()
    file_name = os.path.splitext(os.path.basename(doc_path))[0]
    logger.info(f"File name: {file_name}")
    return file_name, doc


def write_model_as_json(model: BaseModel, file_path: str):
    """
    Write a extracted Pydantic model object as a JSON file to the given file path.
    """
    logger.info("Writing the extraction result to a JSON file")
    # Check if the file directory exists, if not, create it.
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(file_path, "w") as f:
        f.write(model.json(indent=4))

    logger.info(f"Saved at {file_path}")


def cosine_similarity(v1, v2) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
