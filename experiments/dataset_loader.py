import dataclasses
import json

import loguru
import tqdm


@dataclasses.dataclass
class SquadQuestion:
    question: str
    question_id: str
    title: str
    context: str


class SquadDatasetLoader:
    def __init__(self) -> None:
        super().__init__()

    def load(self, path):
        pass

    @staticmethod
    def process(file_path: str) -> list[SquadQuestion]:
        """
        Parse the SQuAD dataset into a list of tuples
        """
        loguru.logger.info("Parsing SQuAD dataset")

        with open(file_path, "r") as f:
            squad_data = json.load(f)

        parsed_data = []
        squad_data_tqdm = tqdm.tqdm(squad_data["data"], desc="Parsing SQuAD data")
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
