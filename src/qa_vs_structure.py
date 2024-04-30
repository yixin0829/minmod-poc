import datetime
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import ollama
from loguru import logger
from tqdm import tqdm

# remove the old handler. Else, the old one will work along with the new one you've added below'
logger.remove()
logger.add(sys.stdout, level="INFO")


class Methods(str, Enum):
    ONE_SHOT_QA = "one_shot_qa"
    ONE_SHOT_STRUCTURED = "one_shot_structured"
    ONE_SHOT_STRUCTURED_W_RETRY = "one_shot_structured_w_retry"


@dataclass
class SquadQuestion:
    question: str
    question_id: str
    title: str
    context: str


# Prompts
SYSTEM_PROMPT_QA = """You are a QA bot. Given the context and question, output the final answer only. Here's one example:

```
Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.

Question: Who was the Norse leader?

Answer: Rollo
```

Note: If the answer is not mentioned in the given context, please answer with an empty string ""."""
USER_PROMPT_QA = """Context: {context}\n\nQuestion: {question}\n\nAnswer: """

SYSTEM_PROMPT_STRUCTURED = """You extract structured data from the given context. Given the context and question, output the final answer only. The final answer should be formatted as a JSON instance that conforms to the JSON schema provided. Here's one example:

```
Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.

JSON Schema: {"properties": {"answer": {"default": "", "description": "Who was the Norse leader?", "title": "Answer", "type": "string"}, "required": ["answer"], "title": "Answer", "type": "object"}

Answer: {"answer": "Rollo"}
```

Note: If the answer is not mentioned in the given context, please answer with an empty string ""."""
USER_PROMPT_STRUCTURED = """Context: {context}\n\nJSON Schema: {{"properties": {{"answer": {{"default:": "", "description": "{question}", "title": "Answer", "type": "string"}}, "required": ["answer"], "title": "Answer", "type": "object"}}\n\nAnswer: """


# SQuAD dataset paths
SQUAD_DEV_PATH = "data/asset/eval/squad_v2/dev-v2.0.json"
SQUAD_DEV_TEST_PATH = "data/asset/eval/squad_v2/dev-v2.0_test.json"
OUTPUT_PRED_PATH = "data/asset/eval/squad_v2"

# Note: choose your method here (only need to change this line)
SELECTED_METHOD = Methods.ONE_SHOT_STRUCTURED_W_RETRY

OUTPUT_JSON_NAME = "{}_pred_llama3_8b_{}.json".format(
    datetime.datetime.now().strftime("%Y%m%d%H%M"), SELECTED_METHOD.value
)
RETRY = True if SELECTED_METHOD == Methods.ONE_SHOT_STRUCTURED_W_RETRY else False


class SquadQA(object):
    def __init__(self) -> None:
        self.model = "llama3-custom"

    @staticmethod
    def split_squad(file_path: str, split_ratio: float = 0.8) -> None:
        """Split the SQuAD dataset into train and test sets based on the split ratio"""

        logger.info("Splitting SQuAD dataset into train and test sets")

        with open(file_path, "r") as f:
            squad_data = json.load(f)

        total_data = len(squad_data["data"])
        split_index = int(total_data * split_ratio)

        train_data = squad_data["data"][:split_index]
        test_data = squad_data["data"][split_index:]

        train_data_path = file_path.replace(".json", "_train.json")
        test_data_path = file_path.replace(".json", "_test.json")

        logger.info(f"Writing {len(train_data)} train data to {train_data_path}")
        with open(train_data_path, "w") as f:
            json.dump({"data": train_data}, f)

        logger.info(f"Writing {len(test_data)} test data to {test_data_path}")
        with open(test_data_path, "w") as f:
            json.dump({"data": test_data}, f)

    @staticmethod
    def parse_squad(file_path: str) -> list[SquadQuestion]:
        """Parse the SQuAD dataset into a list of tuples containing (question, question_id, title, context, gt_answer)"""
        # create a named tuple to store the parsed data

        logger.info("Parsing SQuAD dataset")

        with open(file_path, "r") as f:
            squad_data = json.load(f)

        parsed_data = []
        # Note: comment this line to test the code on one document
        # squad_data_tqdm = tqdm(squad_data["data"][:1], desc="Parsing SQuAD data")
        squad_data_tqdm = tqdm(squad_data["data"], desc="Parsing SQuAD data")
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

    def predict(self, question: str, context: str, method: Methods) -> str | None:
        """Predict the answer for a given question and context"""

        if method == Methods.ONE_SHOT_QA:
            sys_prompt = SYSTEM_PROMPT_QA
            user_prompt = USER_PROMPT_QA.format(context=context, question=question)
        elif method in [
            Methods.ONE_SHOT_STRUCTURED,
            Methods.ONE_SHOT_STRUCTURED_W_RETRY,
        ]:
            sys_prompt = SYSTEM_PROMPT_STRUCTURED
            user_prompt = USER_PROMPT_STRUCTURED.format(
                context=context, question=question
            )
        else:
            raise ValueError(f"Method {method} is not supported")

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except ollama.ResponseError as e:
            logger.error(e)
            return None

        answer = response["message"]["content"]
        logger.debug(f"{sys_prompt=}")
        logger.debug(f"{user_prompt=}")
        logger.debug(f"{answer=}")

        return answer

    def bulk_predict(
        self, parsed_data: list[SquadQuestion], method: Methods, retry: bool
    ) -> dict:
        """Predict the answers for a list of questions and contexts."""

        logger.info(
            "Bulk predicting answers for SQuAD dataset use method: {}".format(method)
        )

        # init trackers for answers, summary, and retry distribution
        answers, summary, retry_distribution = {}, defaultdict(int), defaultdict(int)

        parsed_data_tqdm = tqdm(parsed_data, desc="Bulk predicting answers")
        for data in parsed_data_tqdm:
            question = data.question
            context = data.context
            question_id = data.question_id
            answer = self.predict(question, context, method)

            # parse and append to answers list based on the method
            if answer and method == Methods.ONE_SHOT_QA:
                answers[question_id] = answer
                summary["success"] += 1
            elif answer and method in [
                Methods.ONE_SHOT_STRUCTURED,
                Methods.ONE_SHOT_STRUCTURED_W_RETRY,
            ]:
                # empty string is a valid answer for structured prediction
                if not answer:
                    answers[question_id] = ""

                # regex match only the JSON string
                answer_regex = re.match(r"\{.*\}", answer)
                answer = answer_regex.group(0) if answer_regex else answer

                # validate if the answer is a valid JSON string and parse the answer
                try:
                    answer = json.loads(answer)
                    answers[question_id] = answer["answer"]
                    summary["success"] += 1
                except json.JSONDecodeError:
                    logger.error(
                        f"JSON decode error. Skip question_id: {question_id}. {answer=}"
                    )
                    answers[question_id] = None
                    summary["json_parsing_error"] += 1
                except KeyError:
                    logger.error(
                        f"Key error. skip question_id: {question_id}. {answer=}"
                    )
                    answers[question_id] = None
                    summary["key_error"] += 1
                except TypeError:
                    logger.error(
                        f"Answer is type {type(answer)}. Skip question_id: {question_id}. {answer=}"
                    )
                    answers[question_id] = None
                    summary["type_error"] += 1

            # retry the prediction if the answer is None for up to N times
            if retry and answers[question_id] is None:
                retry_count = 0
                while retry_count < 10:
                    logger.warning(
                        "Retrying prediction for question_id: {}".format(question_id)
                    )
                    answer = self.predict(question, context, method)
                    summary["retry"] += 1
                    retry_distribution[question_id] += 1
                    try:
                        answer = json.loads(answer)
                        answers[question_id] = answer["answer"]
                        logger.info(
                            "Prediction successful after retry for question_id: {}".format(
                                question_id
                            )
                        )
                        summary["retry_success"] += 1
                        break
                    except Exception as e:
                        logger.error(f"{answer=}. {e}")
                        retry_count += 1

            parsed_data_tqdm.refresh()

        # bulk prediction summary and relative percentage
        logger.info(f"Summary: {summary}")
        return answers


if __name__ == "__main__":
    """Run the SquadQA class to predict answers for the SQuAD dataset. Finally write predictions to a JSON file."""
    squad_qa = SquadQA()

    parsed_data = squad_qa.parse_squad(SQUAD_DEV_TEST_PATH)
    answers = squad_qa.bulk_predict(parsed_data, method=SELECTED_METHOD, retry=RETRY)

    logger.info("Writing predictions to JSON file at {}".format(OUTPUT_PRED_PATH))
    with open(os.path.join(OUTPUT_PRED_PATH, OUTPUT_JSON_NAME), "w") as f:
        json.dump(answers, f)
