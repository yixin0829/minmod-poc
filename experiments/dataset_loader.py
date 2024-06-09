import dataclasses
import json

import tqdm
from loguru import logger


@dataclasses.dataclass
class SquadQuestion:
    question: str
    question_id: str
    title: str
    context: str


class SquadDatasetLoader:
    def __init__(self) -> None:
        super().__init__()

    def load(self, path: str) -> list[SquadQuestion]:
        logger.info(f"Loading SQuAD dataset from {path}")
        with open(path, "r") as f:
            data = json.load(f)

        data_processed = self._process(data)
        return data_processed

    @staticmethod
    def _process(data: dict) -> list[SquadQuestion]:
        logger.info("Parsing SQuAD dataset into a list of tuples")
        parsed_data = []
        squad_data_tqdm = tqdm.tqdm(data["data"], desc="Parsing SQuAD data")

        for data in squad_data_tqdm:
            title = data["title"]
            for paragraph in data["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    question_id = qa["id"]
                    parsed_data.append(
                        SquadQuestion(question, question_id, title, context)
                    )

            squad_data_tqdm.refresh()

        return parsed_data
