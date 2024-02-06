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


# Max context window of GPT-4-turbo is 128,000 tokens.
print(count_file_tokens("data/asset/parsed_result/Bleiberg_Pb_Zn_5-2017/result.txt"))
print(
    count_file_tokens("data/asset/parsed_result/Bongará_Zn_3-2019/result.txt")
)  # 73370 tokens ~= 200 pages
