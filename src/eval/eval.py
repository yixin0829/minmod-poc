import json
import os

from langsmith import Client
from langsmith.schemas import Example, Run


class MinModEvaluator:
    def __init__(self) -> None:
        pass

    def create_dataset(self):
        """
        Create a dataset on LangSmith for evaluating the MinMod extraction model on the MineralSite schema
        """

        # Inputs are provided to your model, so it know what to generate
        dataset_inputs = []
        # Outputs are provided to the evaluator, so it knows what to compare to
        dataset_outputs = []

        # Read the txt files from parsed_pdf_w_gt directory and append to input list, then find corresponding output from simplified ground truth directory and append to output list
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
