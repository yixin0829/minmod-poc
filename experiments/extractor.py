import collections
import json
import random
import re
import time

import tqdm
from dataset_loader import SquadDatasetLoader
from llm import BaseLLM
from loguru import logger
from prompts import PromptFactory


class Extractor:
    def __init__(
        self,
        model: BaseLLM,
        data_loader: SquadDatasetLoader,
        prompt_factory: PromptFactory,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.prompt_factory = prompt_factory

    @staticmethod
    def _add_group_answers_to_answers(
        context: str,
        question_ids: list[str],
        questions: list[str],
        response: str,
        group_answers: list[str],
        answers: dict,
        summary: dict,
        failure_log: dict,
    ):
        """
        Add the group answers to master answers dict for eval + error handling
        answers, summary, failure_log are mutable objects that will be updated in place in parent function
        """
        logger.info(f"{group_answers=}")
        if len(group_answers) == len(question_ids):
            # SQuAD map unanswerable questions to "" so we do the same
            for id, answer in zip(question_ids, group_answers):
                try:
                    answers[id] = "" if "N/A" in answer else answer
                    summary["successful_answered_qs"] += 1

                except Exception as e:
                    logger.error(
                        f"Error in loading group_answers to answers. {e=}\n{id=}\n{answer=}\nFallback to append exception to master answers dict"
                    )
                    summary["append_answers_error"] += 1
                    answers[id] = str(e)
                    failure_log.append(
                        {
                            "context": context,
                            "questions": questions,
                            "response": response,
                            "group_answers": group_answers,
                            "reason": f"Error in loading group_answers to answers. {e=}",
                        }
                    )
            summary[f"batch_size_{len(questions)}_success"] += 1
        else:
            summary["mismatched_group_answers_len_error"] += 1
            failure_log.append(
                {
                    "context": context,
                    "questions": questions,
                    "response": response,
                    "group_answers": group_answers,
                    "reason": "Mismatched group answers length",
                }
            )

    @staticmethod
    def _add_experiment_stats(start_time, end_time, answers, summary):
        summary["time_taken"] = end_time - start_time
        summary["avg_time_per_q"] = summary["time_taken"] / len(answers)
        summary["total_fields"] = len(answers)
        logger.info(f"{summary=}")

    @staticmethod
    def _parse_qa_response(
        response: str, context, questions: list[str], summary: dict, failure_log: dict
    ):
        # parse llm response to a list of answers for this question group
        group_answers = []
        for i in range(len(questions)):
            try:
                pat = re.compile(f"A\[{i+1}\]: (.*)")
                match = re.search(pat, response)
                group_answers.append(
                    match.group(1).replace(f"A[{i+1}]: ", "") if match else ""
                )
            except Exception as e:
                logger.error(
                    f"Error in parsing response. {e=}\n{pat=}\n{response=}\nFallback to append exception to group_answers"
                )
                summary["parse_error"] += 1
                group_answers.append(str(e))
                failure_log.append(
                    {
                        "context": context,
                        "questions": questions,
                        "response": response,
                        "group_answers": group_answers,
                        "reason": f"Error in parsing response {e=}",
                    }
                )
        return group_answers

    def _parse_structured_response(
        self,
        response: str,
        context,
        questions: list[str],
        summary: dict,
        failure_log: dict,
    ):
        # parse llm response for this group of question and append to answers list
        group_answers = []
        try:
            response_processed = response.replace("\n", "")
            # match = re.match(r"\{.*\}", answer)
            # answer = match.group(0) if match else answer
            curly_start = response_processed.find("{")
            curly_end = response_processed.rfind("}")

            # find the json string in the response
            response_processed = response_processed[curly_start : curly_end + 1]
            response_dict = json.loads(response_processed)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error. {response=}")
            summary["json_decode_error"] += 1
            failure_log.append(
                {
                    "context": context,
                    "questions": questions,
                    "response": response,
                    "group_answers": group_answers,
                    "reason": f"Error in parsing JSON string. {e=}",
                }
            )

        for i in range(len(questions)):
            try:
                ans = response_dict[f"A[{i+1}]"]
            except KeyError as e:
                logger.error(f"Key error. {response=}")
                summary["key_error"] += 1
                ans = "Key Error"
                failure_log.append(
                    {
                        "context": context,
                        "questions": questions,
                        "response": response,
                        "group_answers": group_answers,
                        "reason": f"Error in accessing parsed JSON object. {e=}",
                    }
                )

            group_answers.append(ans)

        return group_answers

    def batch_qa(self, batch_size: int, dataset_path: str):
        logger.info(f"Batch QA for {dataset_path=} with {batch_size=}")
        dataset = self.data_loader.load(dataset_path)

        # group questions by context
        context_qs_map = collections.defaultdict(list)
        for data in dataset:
            context_qs_map[data.context].append(data)

        # shuffle the questions to avoid any bias with the same seed
        for context, qs_group in context_qs_map.items():
            random.Random(1).shuffle(qs_group)

        # init trackers for experiment report
        answers = {}
        summary = collections.defaultdict(int)
        failure_log = []
        start_time = time.time()

        # loop through each context in dataset and group questions by batch_size
        context_qs_map = tqdm.tqdm(context_qs_map.items(), desc="Batch QA")
        for context, qs_group in context_qs_map:
            logger.info(f"{summary=}")
            num_questions = len(qs_group)
            num_groups = num_questions // batch_size

            # if the number of questions is not divisible by batch_size, add one more group to handle the tail
            if num_questions % batch_size != 0:
                num_groups += 1

            for i in range(num_groups):
                start = i * batch_size
                end = (i + 1) * batch_size

                # if end is greater than the number of questions, set end to the tail of the list
                if end > num_questions:
                    end = num_questions

                questions = [data.question for data in qs_group[start:end]]
                question_ids = [data.question_id for data in qs_group[start:end]]

                messages = self.prompt_factory.batch_qa_messages(questions, context)
                response = self.model.chat(
                    messages=messages,
                    json_mode=False,
                )

                # Note: debugging print
                # print("----")
                # for message in messages:
                #     print(message)
                # print("----")
                # print(response)
                # print("----")

                group_answers = self._parse_qa_response(
                    response, context, questions, summary, failure_log
                )
                self._add_group_answers_to_answers(
                    context,
                    question_ids,
                    questions,
                    response,
                    group_answers,
                    answers,
                    summary,
                    failure_log,
                )

        self._add_experiment_stats(start_time, time.time(), answers, summary)

        return answers, summary, failure_log

    def batch_structured(self, batch_size: int, dataset_path: str):
        logger.info(
            f"Batch structured extraction for {dataset_path=} with {batch_size=}"
        )
        dataset = self.data_loader.load(dataset_path)

        # group questions by context
        context_qs_map = collections.defaultdict(list)
        for data in dataset:
            context_qs_map[data.context].append(data)

        # shuffle the questions to avoid any bias with the same seed
        for context, qs_group in context_qs_map.items():
            random.Random(1).shuffle(qs_group)

        # init trackers for experiment report
        answers = {}
        summary = collections.defaultdict(int)
        failure_log = []
        start_time = time.time()

        # loop through dataset and group questions by batch_size
        context_qs_map = tqdm.tqdm(
            context_qs_map.items(), desc="Batch Structured Extraction"
        )
        for context, qs_group in context_qs_map:
            logger.info(f"{summary=}")
            num_questions = len(qs_group)
            num_groups = num_questions // batch_size

            # if the number of questions is not divisible by batch_size, add one more group to handle the tail
            if num_questions % batch_size != 0:
                num_groups += 1

            for i in range(num_groups):
                start = i * batch_size
                end = (i + 1) * batch_size

                # if end is greater than the number of questions, set end to the tail of the list
                if end > num_questions:
                    end = num_questions

                questions = [data.question for data in qs_group[start:end]]
                question_ids = [data.question_id for data in qs_group[start:end]]

                messages = self.prompt_factory.batch_structured_messages(
                    questions, context
                )
                response = self.model.chat(messages=messages, json_mode=True)

                # Note: debugging print
                # print("----")
                # for message in messages:
                #     print(message)
                # print("----")
                # print(response)
                # print("----")

                group_answers = self._parse_structured_response(
                    response, context, questions, summary, failure_log
                )
                self._add_group_answers_to_answers(
                    context,
                    question_ids,
                    questions,
                    response,
                    group_answers,
                    answers,
                    summary,
                    failure_log,
                )

        self._add_experiment_stats(start_time, time.time(), answers, summary)

        return answers, summary, failure_log

    def batch_qa_structured(self, batch_size: int, dataset_path: str):
        logger.info(
            f"Batch QA -> structured extraction for {dataset_path=} with {batch_size=}"
        )
        dataset = self.data_loader.load(dataset_path)

        # group questions by context
        context_qs_map = collections.defaultdict(list)
        for data in dataset:
            context_qs_map[data.context].append(data)

        # shuffle the questions to avoid any bias with the same seed
        for context, qs_group in context_qs_map.items():
            random.Random(1).shuffle(qs_group)

        # init trackers for experiment report
        answers = {}
        summary = collections.defaultdict(int)
        failure_log = []
        start_time = time.time()

        # loop through each context in dataset and group questions by batch_size
        context_qs_map = tqdm.tqdm(
            context_qs_map.items(), desc="Batch QA -> Structured Extraction"
        )
        for context, qs_group in context_qs_map:
            logger.info(f"{summary=}")
            num_questions = len(qs_group)
            num_groups = num_questions // batch_size

            # if the number of questions is not divisible by batch_size, add one more group to handle the tail
            if num_questions % batch_size != 0:
                num_groups += 1

            for i in range(num_groups):
                start = i * batch_size
                end = (i + 1) * batch_size

                # if end is greater than the number of questions, set end to the tail of the list
                if end > num_questions:
                    end = num_questions

                questions = [data.question for data in qs_group[start:end]]
                question_ids = [data.question_id for data in qs_group[start:end]]

                messages = self.prompt_factory.batch_qa_messages(questions, context)
                response = self.model.chat(
                    messages=messages,
                    json_mode=False,
                )

                # feed the response as part of the context for structured extraction
                messages = self.prompt_factory.batch_qa_structured_messages(
                    questions, response, context
                )
                response = self.model.chat(messages=messages, json_mode=True)

                # Note: debugging print
                print("----")
                for message in messages:
                    print(message)
                print("----")
                print(response)
                print("----")

                group_answers = self._parse_structured_response(
                    response, context, questions, summary, failure_log
                )
                self._add_group_answers_to_answers(
                    context,
                    question_ids,
                    questions,
                    response,
                    group_answers,
                    answers,
                    summary,
                    failure_log,
                )

        self._add_experiment_stats(start_time, time.time(), answers, summary)

        return answers, summary, failure_log
