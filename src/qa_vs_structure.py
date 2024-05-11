import argparse
import datetime
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import ollama
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, create_model
from tqdm import tqdm

import config.prompts as prompts
from config.config import Config, EmbeddingFunction
from utils.utils import cosine_similarity

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# remove the old handler. Else, the old one will work along with the new one you've added below'
logger.remove()
# Note: Config loguru logger to log to console and file
LOG_LEVEL = "INFO"
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
logger.add(sys.stdout, level=LOG_LEVEL)


class Methods(str, Enum):
    SINGLE_QA = "1shot_qa"
    SINGLE_STRUCTURED = "2shot_structured"
    SINGLE_STRUCTURED_W_RETRY = "2shot_structured_w_retry"
    MULTI_FIELD_QA = "1shot_multi_field_qa"
    MULTI_FIELD_STRUCTURED = "1shot_multi_field_structured"


@dataclass
class SquadQuestion:
    question: str
    question_id: str
    title: str
    context: str


class SquadQA(object):
    def __init__(self, model: str, fixed_samples: bool) -> None:
        self.model = model
        self.fixed_samples = fixed_samples

    @staticmethod
    def enrich_squad(file_path: str) -> None:
        """
        Enrich the SQuAD dataset by rewriting question into a description field
        """

        logger.info("Enriching SQuAD dataset with descriptions")

        # load the SQuAD dataset
        with open(file_path, "r") as f:
            squad_data = json.load(f)

        squad_data_tqdm = tqdm(squad_data["data"], desc="Enriching SQuAD data")
        for data in squad_data_tqdm:
            for paragraph in data["paragraphs"]:
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    try:
                        response = ollama.chat(
                            model="llama3",
                            messages=[
                                {
                                    "role": "system",
                                    "content": prompts.FEW_SHOT_ENRICH_SYS_PROMPT,
                                },
                                {
                                    "role": "user",
                                    "content": prompts.FEW_SHOT_ENRICH_USER_PROMPT.format(
                                        question=question.strip()
                                    ),
                                },
                            ],
                        )
                        description = response["message"]["content"]
                        # remove "Description: " from the response to safe guard against any errors
                        description = description.replace("Description: ", "")
                        description = description.replace("Input: ", "")
                        # capitalize the first letter of the description
                        description = description.capitalize().strip()
                        # compute the cosine similarity between the question and generated description
                        embed_q = client.embeddings.create(
                            input=question,
                            model=EmbeddingFunction.TEXT_EMBEDDING_3_SMALL.value,
                        )
                        embed_d = client.embeddings.create(
                            input=description,
                            model=EmbeddingFunction.TEXT_EMBEDDING_3_SMALL.value,
                        )
                        similarity = cosine_similarity(
                            embed_q.data[0].embedding, embed_d.data[0].embedding
                        )

                        MAX_RETRY = 10
                        retry_count = 0
                        # Note: Evaluating quality based on cosine similarity is not the best approach. Sometimes opposite words can have high similarity.
                        while similarity < 0.75 and retry_count < MAX_RETRY:
                            # regenerate the description if the similarity is less than 0.95
                            response = ollama.chat(
                                model="llama3-custom",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": prompts.FEW_SHOT_ENRICH_SYS_PROMPT,
                                    },
                                    {
                                        "role": "user",
                                        "content": prompts.FEW_SHOT_ENRICH_USER_PROMPT.format(
                                            question=question.strip()
                                        ),
                                    },
                                ],
                            )
                            description_temp = response["message"]["content"]
                            description_temp = description_temp.replace(
                                "Description: ", ""
                            )
                            description_temp = description_temp.replace("Input: ", "")
                            description_temp = description_temp.capitalize().strip()
                            # compute the cosine similarity between the question and generated description
                            embed_d_temp = client.embeddings.create(
                                input=description_temp,
                                model=EmbeddingFunction.TEXT_EMBEDDING_3_SMALL.value,
                            )
                            similarity_temp = cosine_similarity(
                                embed_q.data[0].embedding,
                                embed_d_temp.data[0].embedding,
                            )

                            # take the new description if the similarity is higher
                            if similarity_temp > similarity:
                                description = description_temp
                                similarity = similarity_temp
                                logger.info(
                                    f"higher similarity found: {similarity_temp}"
                                )

                            retry_count += 1

                        logger.info(f"{question=}, {description=}")
                        qa["description"] = description
                    except ollama.ResponseError as e:
                        qa["description"] = ""
                        logger.error(e)

            squad_data_tqdm.refresh()

        # write the enriched SQuAD dataset to a new file
        enriched_file_path = file_path.replace(".json", "_enriched.json")
        logger.info(f"Writing enriched SQuAD data to {enriched_file_path}")
        with open(enriched_file_path, "w") as f:
            json.dump(squad_data, f)

        logger.info("Enrichment completed")

    @staticmethod
    def split_squad(file_path: str, split_ratio: float = 0.8) -> None:
        """
        Split the SQuAD dataset into train and test sets based on the split ratio
        """

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
        """
        Parse the SQuAD dataset into a list of tuples containing (question, question_id, title, context, gt_answer)
        """
        logger.info("Parsing SQuAD dataset")

        with open(file_path, "r") as f:
            squad_data = json.load(f)

        parsed_data = []
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

    def predict(self, question: str, context: str, method: Methods) -> str:
        """Predict the answer for a given question and context"""

        answer = ""

        # construct prompts
        if method == Methods.SINGLE_QA:
            sys_prompt = prompts.SINGLE_QA_SYS_PROMPT
            user_prompt = prompts.SINGLE_QA_USER_PROMPT.format(
                context=context, question=question.strip()
            )
        elif method in [
            Methods.SINGLE_STRUCTURED,
            Methods.SINGLE_STRUCTURED_W_RETRY,
        ]:
            sys_prompt = prompts.SINGLE_STRUCTURED_SYS_PROMPT

            # construct a pydantic model in run time for generating the JSON schema
            Answer = create_model(
                "Answer",
                answer=(str, Field(default="N/A", description=question.strip())),
            )
            json_schema = json.dumps(Answer.model_json_schema())

            user_prompt = prompts.SINGLE_STRUCTURED_USER_PROMPT.format(
                context=context, json_schema=json_schema
            )

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

        answer = response["message"]["content"]
        logger.debug(f"{sys_prompt=}")
        logger.debug(f"{user_prompt=}")
        logger.info(f"{answer=}")

        return answer

    @staticmethod
    def stratified_sample_qs(
        squad_data: dict, num_questions: int, fixed_samples: bool
    ) -> list[dict]:
        """
        Sample questions from the SQuAD dataset for building few-shot prompts
        Args:
            squad_data: a SQuAD dataset
            num_questions: number of questions to sample
            fixed_samples: whether to sample fixed questions for few-shot prompts
        Returns:
            sample_qs: a list of sampled question objects from SQaUD dataset
        """
        # Note: initial experiments show that sampling same number of questions for few-shot prompting does not improve the performance
        # Using one-shot example with 2 questions (one w answer + one w/o answer) yields the same result

        # stratified sample same number of answerable/unanswerable questions to use in few-shot examples
        questions = squad_data["data"][0]["paragraphs"][0]["qas"]
        df = pd.DataFrame(questions)
        if not fixed_samples:
            df = df.groupby("is_impossible", group_keys=False)[
                ["question", "id", "answers"]
            ].apply(lambda x: x.sample(num_questions // 2, replace=True))
        else:
            df = df.groupby("is_impossible", group_keys=False)[
                ["question", "id", "answers"]
            ].apply(lambda x: x.sample(1))

        # shuffle the dataframe to avoid bias in ordering in the few-shot prompt
        df = df.sample(frac=1).reset_index(drop=True)

        sample_qs = df.to_dict(orient="records")

        return sample_qs

    def predict_multi_fields(
        self, questions: list[str], context: str, method: Methods
    ) -> str:
        """
        Predict the answers for a list of questions and a single context
        """
        num_questions = len(questions)

        if method == Methods.MULTI_FIELD_QA:
            # construct the system prompt
            sys_prompt = prompts.MULTI_FIELD_QA_SYS_PROMPT
            with open(SQUAD_DEV_TRAIN_PATH, "r") as f:
                squad_data = json.load(f)
                if num_questions == 1:
                    sample_qs = self.stratified_sample_qs(
                        squad_data, 2, self.fixed_samples
                    )
                else:
                    sample_qs = self.stratified_sample_qs(
                        squad_data, num_questions, self.fixed_samples
                    )

                sample_qs_concat = ""
                sample_ans_concat = ""
                for i in range(len(sample_qs)):
                    q = sample_qs[i]["question"]
                    if sample_qs[i]["answers"]:
                        ans = sample_qs[i]["answers"][0]["text"]
                    else:
                        ans = "N/A"
                    sample_qs_concat += f"Question{i+1}: {q}\n"
                    sample_ans_concat += f"Answer{i+1}: {ans}\n"

                sys_prompt = prompts.MULTI_FIELD_QA_SYS_PROMPT.format(
                    questions=sample_qs_concat, answers=sample_ans_concat
                )

            # construct the user prompt
            questions_concat = ""
            for i, q in enumerate(questions):
                questions_concat += f"Question{i+1}: {q}\n"

            user_prompt = prompts.MULTI_FIELD_QA_USER_PROMPT.format(
                context=context, questions=questions_concat
            )
        elif method == Methods.MULTI_FIELD_STRUCTURED:
            # construct the system prompt
            sys_prompt = prompts.MULTI_FIELD_STRUCTURED_SYS_PROMPT
            with open(SQUAD_DEV_TRAIN_PATH, "r") as f:
                squad_data = json.load(f)
                if num_questions == 1:
                    sample_qs = self.stratified_sample_qs(
                        squad_data, 2, self.fixed_samples
                    )
                else:
                    sample_qs = self.stratified_sample_qs(
                        squad_data, num_questions, self.fixed_samples
                    )

                schema_fields = {}
                ans_fields = {}
                for i, qas in enumerate(sample_qs):
                    q = qas["question"]
                    schema_fields[f"answer{i+1}"] = (
                        str,
                        Field(default="N/A", description=q),
                    )
                    if qas["answers"]:
                        ans_fields[f"answer{i+1}"] = sample_qs[i]["answers"][0]["text"]
                    else:
                        ans_fields[f"answer{i+1}"] = "N/A"
                Answer = create_model("Answer", **schema_fields)
                sys_prompt = prompts.MULTI_FIELD_STRUCTURED_SYS_PROMPT.format(
                    json_schema=json.dumps(Answer.model_json_schema()),
                    answers=json.dumps(ans_fields),
                )

            # construct the user prompt
            schema_fields_user = {}
            for i, q in enumerate(questions):
                schema_fields_user[f"answer{i+1}"] = (
                    str,
                    Field(default="N/A", description=q),
                )
            Answer_user = create_model("Answer", **schema_fields_user)
            user_prompt = prompts.MULTI_FIELD_STRUCTURED_USER_PROMPT.format(
                context=context, json_schema=json.dumps(Answer_user.model_json_schema())
            )

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

        # parse the answer string to get the answers
        answer = response["message"]["content"]
        if LOG_LEVEL == "DEBUG":
            print(sys_prompt)
            print(user_prompt)
        # logger.debug(f"{sys_prompt=}")
        # logger.debug(f"{user_prompt=}")
        logger.debug(f"{answer=}")

        return answer

    def bulk_predict(
        self, parsed_data: list[SquadQuestion], method: Methods, retry: bool
    ):
        """
        Predict the answers for a list of questions and contexts.
        """

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
            if answer and method == Methods.SINGLE_QA:
                answers[question_id] = "" if "N/A" in answer else answer
                summary["success"] += 1
            elif answer and method in [
                Methods.SINGLE_STRUCTURED,
                Methods.SINGLE_STRUCTURED_W_RETRY,
            ]:
                # regex match only the JSON string. re.DOTALL is used to match newline characters
                match = re.match(r"\{.*\}", answer, re.DOTALL)
                answer = match.group(0) if match else answer

                try:
                    answer = json.loads(answer)["answer"]
                    answers[question_id] = "" if "N/A" in answer else answer
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
                MAX_RETRY = 10
                while retry_count < MAX_RETRY:
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
                    except json.JSONDecodeError as e:
                        logger.error(f"{e}\n{answer=}")
                        retry_count += 1

            parsed_data_tqdm.refresh()

        # bulk prediction summary and relative percentage
        logger.info(f"Summary: {summary}")
        return answers, summary

    def bulk_predict_multi_fields(
        self,
        parsed_data: list[SquadQuestion],
        method: Methods,
        retry: bool,
        group_size: int,
    ):
        """Predict the answers for a list of questions and contexts."""

        logger.info(
            "Bulk predicting answers for SQuAD dataset use method: {}".format(method)
        )

        # group questions by context
        grouped_qs = defaultdict(list)
        for data in parsed_data:
            grouped_qs[data.context].append(data)

        # init trackers for answers, summary, and retry distribution
        answers, summary = {}, defaultdict(int)

        # loop through d and group questions by group_size. if the questions are from different context then split them into different groups
        grouped_qs = tqdm(grouped_qs.items(), desc="Bulk predicting answers")
        for context, group in grouped_qs:
            num_questions = len(group)
            num_groups = num_questions // group_size
            summary[f"number_of_groups_{group_size}"] += num_groups
            if num_questions % group_size != 0:
                num_groups += 1
                summary[f"number_of_incomplete_groups"] += 1

            for i in range(num_groups):
                answers_group = []
                start = i * group_size
                end = (i + 1) * group_size

                # if end is greater than the number of questions, set end to the tail of the list
                if end > num_questions:
                    end = num_questions

                questions = [data.question for data in group[start:end]]
                question_ids = [data.question_id for data in group[start:end]]

                answer = self.predict_multi_fields(questions, context, method)

                # @ parse and append to answers list
                if method == Methods.MULTI_FIELD_QA:
                    for i in range(len(questions)):
                        match = re.search(f"Answer{i+1}: (.*)", answer)
                        answers_group.append(
                            match.group(1).replace(f"Answer{i+1}: ", "")
                            if match
                            else ""
                        )
                elif method == Methods.MULTI_FIELD_STRUCTURED:
                    try:
                        answer = answer.replace("\n", "")
                        # match = re.match(r"\{.*\}", answer)
                        # answer = match.group(0) if match else answer
                        curly_start = answer.find("{")
                        curly_end = answer.rfind("}") + 1
                        answer = answer[curly_start:curly_end]
                        answer = json.loads(answer)
                        for i in range(len(questions)):
                            answers_group.append(answer[f"answer{i+1}"])
                    except json.JSONDecodeError:
                        logger.error(f"JSON decode error. {answer=}")
                    except KeyError:
                        logger.error(f"Key error. {answer=}")

                # retry the prediction if the answer group is empty
                if not answers_group:
                    retry_count = 0
                    MAX_RETRY = 10
                    while retry_count < MAX_RETRY:
                        logger.warning(
                            "Retrying prediction for question_ids: {}".format(
                                question_ids
                            )
                        )
                        answer = self.predict_multi_fields(questions, context, method)
                        summary["retry"] += 1
                        try:
                            # note: replace newline characters to avoid JSONDecodeError
                            answer = answer.replace("\n", "")
                            curly_start = answer.find("{")
                            curly_end = answer.rfind("}") + 1
                            answer = answer[curly_start:curly_end]
                            # match = re.match(r"\{.*\}", answer)
                            # answer = match.group(0) if match else answer
                            answer = json.loads(answer)
                            for i in range(len(questions)):
                                answers_group.append(answer[f"answer{i+1}"])
                            logger.info(
                                "Prediction successful after retry for question_ids: {}".format(
                                    question_ids
                                )
                            )
                            summary["retry_success"] += 1
                            break
                        except json.JSONDecodeError as e:
                            logger.error(f"{e}\n{answer=}")
                            retry_count += 1

                logger.info(f"{answers_group=}")
                if answers_group:
                    for id, ans in zip(question_ids, answers_group):
                        answers[id] = "" if "N/A" in ans else ans
                    summary["success"] += 1
                else:
                    # still not answers after retry, default to None
                    summary["failed"] += 1
                    for id in question_ids:
                        answers[id] = None

        # bulk prediction summary and relative percentage
        logger.info(f"Summary: {summary}")
        return answers, summary


if __name__ == "__main__":
    """Run the SquadQA class to predict answers for the SQuAD dataset. Finally write predictions to a JSON file."""

    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use for prediction")
    parser.add_argument("--method", type=Methods, help="Method to use for prediction")
    parser.add_argument(
        "--fixed_samples",
        type=int,
        help="Whether to sample fixed questions for few-shot prompts",
    )
    parser.add_argument("--group_size", type=int, help="Group size for multi-field QA")
    args = parser.parse_args()

    MODEL = args.model
    FIXED_SAMPLES = bool(args.fixed_samples) if args.fixed_samples else False
    SELECTED_METHOD = args.method
    GROUP_SIZE = args.group_size

    # SQuAD dataset paths
    SQUAD_DEV_PATH = "data/asset/eval/squad_v2/dev-v2.0.json"
    SQUAD_DEV_TEST_PATH = "data/asset/eval/squad_v2/dev-v2.0_test.json"
    SQUAD_DEV_TRAIN_PATH = "data/asset/eval/squad_v2/dev-v2.0_train.json"
    OUTPUT_PRED_PATH = "data/asset/eval/squad_v2"

    OUTPUT_JSON_NAME = (
        "{date}_pred_{model}_{method}_gs{gs}_fixed{fixed_samples}.json".format(
            date=datetime.datetime.now().strftime("%Y%m%d%H%M"),
            model=MODEL,
            method=SELECTED_METHOD.value,
            gs=GROUP_SIZE,
            fixed_samples=args.fixed_samples,
        )
    )
    RETRY = True if SELECTED_METHOD == Methods.SINGLE_STRUCTURED_W_RETRY else False

    squad_qa = SquadQA(MODEL, FIXED_SAMPLES)

    parsed_data = squad_qa.parse_squad(SQUAD_DEV_TEST_PATH)

    if SELECTED_METHOD in [
        Methods.SINGLE_QA,
        Methods.SINGLE_STRUCTURED,
        Methods.SINGLE_STRUCTURED_W_RETRY,
    ]:
        answers, summary = squad_qa.bulk_predict(
            parsed_data, method=SELECTED_METHOD, retry=RETRY
        )
    elif SELECTED_METHOD in [
        Methods.MULTI_FIELD_QA,
        Methods.MULTI_FIELD_STRUCTURED,
    ]:
        answers, summary = squad_qa.bulk_predict_multi_fields(
            parsed_data, method=SELECTED_METHOD, retry=RETRY, group_size=GROUP_SIZE
        )

    logger.info("Writing predictions to JSON file at {}".format(OUTPUT_PRED_PATH))
    with open(os.path.join(OUTPUT_PRED_PATH, OUTPUT_JSON_NAME), "w") as f:
        json.dump(answers, f)

    logger.info("Writing summary to a text file at {}".format(OUTPUT_PRED_PATH))
    with open(
        os.path.join(OUTPUT_PRED_PATH, OUTPUT_JSON_NAME.replace(".json", ".txt")), "w"
    ) as f:
        f.write(json.dumps(summary, indent=4))
