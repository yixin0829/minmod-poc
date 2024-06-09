from abc import ABC, abstractmethod

import ollama


class BaseLLM(ABC):
    def __init__(self, model_name, temp, max_token) -> None:
        super().__init__()
        self.model_name = model_name
        self.temp = temp
        self.max_token = max_token

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool) -> str:
        pass

    @abstractmethod
    def chat(self, messages: list[str], json_mode: bool) -> str:
        pass


class OllamaLLM(BaseLLM):
    def __init__(self, model_name, temp, max_token) -> None:
        super().__init__(model_name, temp, max_token)

    def generate(self, prompt: str, json_mode: bool) -> str:
        try:
            response = ollama.generate(
                model=self.model_name, prompt=prompt, format="json" if json_mode else ""
            )
        except ollama.ResponseError as e:
            raise e

        return response["response"]

    def chat(self, messages: list[dict], json_mode: bool) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                format="json" if json_mode else "",
            )
        except ollama.ResponseError as e:
            raise e

        # parse the answer string to get the answers
        response_content = response["message"]["content"]
        return response_content
