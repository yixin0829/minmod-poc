import json
import os
import re

from langchain.evaluation import JsonEditDistanceEvaluator
from langchain.smith import RunEvalConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResults
from langsmith.schemas import Example, Run

import config.prompts as prompts
from config.config import Config, ExtractionMethod
from extractor_minmod.extractor import ExtractorBaseline, ExtractorVectorRetriever
from schema.mineral_site import MineralSite


class BasicInfoEvaluator(RunEvaluator):
    def __init__(self):
        pass

    def evaluate_run(
        self, run: Run, example: Example | None = None
    ) -> EvaluationResult | EvaluationResults:
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None")

        pred = run.outputs["output"].dict()["basic_info"]
        ref = example.outputs["output"]["basic_info"]

        # Lowercase all keys and values
        pred = {k.lower(): v.lower() for k, v in pred.items()}
        ref = {k.lower(): v.lower() for k, v in ref.items()}

        # Evaluate the edit distance between the predicted and reference location info
        evaluator = JsonEditDistanceEvaluator()
        result = evaluator.evaluate_strings(prediction=pred, reference=ref)
        return EvaluationResult(
            key="Basic Info JSON Edit Distance", score=result["score"]
        )


class LocationInfoEvaluator(RunEvaluator):
    def __init__(self):
        pass

    def evaluate_run(
        self, run: Run, example: Example | None = None
    ) -> EvaluationResult | EvaluationResults:
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None")

        # Get the location info from the model output
        pred = run.outputs["output"].dict()["location_info"]
        ref = example.outputs["output"]["location_info"]

        # Sort the keys in the location info
        pred = {k: pred[k] for k in sorted(pred)}
        ref = {k: ref[k] for k in sorted(ref)}

        # Lowercase all keys and values
        pred = {k.lower(): v.lower() for k, v in pred.items() if v}
        ref = {k.lower(): v.lower() for k, v in ref.items() if v}

        # Evaluate the edit distance between the predicted and reference location info
        evaluator = JsonEditDistanceEvaluator()
        result = evaluator.evaluate_strings(prediction=pred, reference=ref)
        return EvaluationResult(
            key="Location Info JSON Edit Distance", score=result["score"]
        )


class MineralInventoryEvaluator(RunEvaluator):
    def __init__(self):
        llm = ChatOpenAI(model=Config.EVAL_MODEL_NAME, temperature=0)
        template = prompts.INVENTORY_EVAL_TEMPLATE

        self.eval_chain = (
            PromptTemplate.from_template(template) | llm | StrOutputParser()
        )

    def evaluate_run(
        self, run: Run, example: Example | None = None
    ) -> EvaluationResult | EvaluationResults:
        pred = run.outputs["output"].dict()["mineral_inventory"]
        ref = example.outputs["output"]["mineral_inventory"]

        evaluator_result = self.eval_chain.invoke({"pred": pred, "ref": ref})

        # Split last line to get the score
        split_response = evaluator_result.split("\n")
        reasoning, score = "\n".join(split_response[:-1]), split_response[-1]
        score = re.search(r"\d+", score).group(0)
        if score is not None:
            score = float(score.strip()) / 100.0
        return EvaluationResult(
            key="Inventory Similarity (LLM Eval)",
            score=score,
            comment=reasoning.strip(),
        )


class DepositTypeEvaluator(RunEvaluator):
    def __init__(self):
        pass

    def evaluate_run(
        self, run: Run, example: Example | None = None
    ) -> EvaluationResult | EvaluationResults:
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None")

        # Get the location info from the model output
        pred = [
            deposit_candidate["observed_name"]
            for deposit_candidate in run.outputs["output"].dict()[
                "deposit_type_candidate"
            ]
        ]
        ref = [
            deposit_candidate["observed_name"]
            for deposit_candidate in example.outputs["output"]["deposit_type_candidate"]
        ]

        # Sort the deposit types
        pred.sort()
        ref.sort()

        # Lowercase all deposit names
        pred = list(map(str.lower, pred))
        ref = list(map(str.lower, ref))

        # Evaluate the edit distance between the predicted and reference location info
        evaluator = JsonEditDistanceEvaluator()
        result = evaluator.evaluate_strings(prediction=pred, reference=ref)
        return EvaluationResult(
            key="Deposit Type JSON Edit Distance", score=result["score"]
        )


class MinModEvaluator:
    def __init__(self) -> None:
        # Evaluation configuration
        self.eval_config = RunEvalConfig(
            evaluators=[],
            custom_evaluators=[
                BasicInfoEvaluator(),
                LocationInfoEvaluator(),
                MineralInventoryEvaluator(),
                DepositTypeEvaluator(),
            ],
        )

        # Init langsmith client
        self.client = Client()

    def create_dataset(self):
        """
        Create a dataset on LangSmith for evaluating the MinMod extraction model on the MineralSite schema
        """

        # Inputs are provided to your model, so it know what to generate
        dataset_inputs = []
        # Outputs are provided to the evaluator, so it knows what to compare to
        dataset_outputs = []

        # Read the txt files from parsed_pdf_w_gt directory and append to input list, then
        # find corresponding output from simplified ground truth directory and append to output list
        for file in os.listdir("data/asset/parsed_pdf_w_gt"):
            with open(f"data/asset/parsed_pdf_w_gt/{file}/{file}.txt", "r") as f:
                dataset_inputs.append({"input": f.read()})

            with open(f"data/asset/ground_truth/simplified/{file}.json", "r") as f:
                data = json.load(f)
                dataset_outputs.append({"output": data})

        client = Client()

        # Storing inputs in a dataset lets us run chains and LLMs over a shared set of examples.
        dataset = client.create_dataset(
            dataset_name="MinMod Extraction Dataset",
            description="Dataset for evaluating MinMod extraction model on MineralSite schema",
        )

        client.create_examples(
            inputs=dataset_inputs,
            outputs=dataset_outputs,
            dataset_id=dataset.id,
        )

    def evaluate(self, dataset_name: str, llm_or_chain_factory):
        if Config.EVAL_METHOD == ExtractionMethod.BASELINE:

            project_metadata = {
                "model": Config.MODEL_NAME,
                "model_temperature": Config.TEMPERATURE,
                "model_max_token": Config.MAX_TOKENS,
                "extraction_method": Config.EVAL_METHOD.value,
            }

            concurrency_level = Config.CONCURRENCY_LEVEL
        elif Config.EVAL_METHOD in [
            ExtractionMethod.VECTOR_RETRIEVER,
            ExtractionMethod.MULTI_QUERY_RETRIEVER,
        ]:
            project_metadata = {
                "model": Config.MODEL_NAME,
                "model_temperature": Config.TEMPERATURE,
                "model_max_token": Config.MAX_TOKENS,
                "chunk_size": Config.CHUNK_SIZE,
                "eval_model": Config.EVAL_MODEL_NAME,
                "extraction_method": Config.EVAL_METHOD.value,
            }

            # Chroma does not support concurrency for vector retriever
            concurrency_level = 1
        else:
            raise ValueError(f"Invalid extraction method: {Config.EVAL_METHOD}")

        self.client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=llm_or_chain_factory,
            evaluation=self.eval_config,
            concurrency_level=concurrency_level,
            project_metadata=project_metadata,
        )


if __name__ == "__main__":
    evaluator = MinModEvaluator()
    config = Config()

    if Config.EVAL_METHOD == ExtractionMethod.BASELINE:
        extractor = ExtractorBaseline(config)
    elif Config.EVAL_METHOD in [
        ExtractionMethod.VECTOR_RETRIEVER,
        ExtractionMethod.MULTI_QUERY_RETRIEVER,
    ]:
        extractor = ExtractorVectorRetriever(config)
    else:
        raise ValueError(f"Invalid extraction method: {Config.EVAL_METHOD}")

    # Option 1: Evalute llm or chain constructor
    # evaluator.evaluate_llm_or_chain(
    #     dataset_name=Config.EVAL_DATASET,
    #     llm_or_chain_factory=extractor_baseline.extraction_chain_factory(MineralSite)
    # )

    # Option 2: Evaluate custom functions
    evaluator.evaluate(
        dataset_name=Config.EVAL_DATASET,
        llm_or_chain_factory=extractor.extract_eval,
    )
