import argparse
import datetime
import enum
import json
import os

from dataset_loader import SquadDatasetLoader
from extractor import Extractor
from llm import OllamaLLM
from loguru import logger
from prompts import PromptFactory


class Methods(str, enum.Enum):
    BATCH_QA = "batch-qa"
    BATCH_STRUCTURED = "batch-structured"
    BATCH_QA_STRUCTURED = "batch-qa-structured"


class Model(str, enum.Enum):
    # Temp = 0.5
    LLAMA3_CUSTOM = "llama3-custom"


if __name__ == "__main__":
    """Run the SquadQA class to predict answers for the SQuAD dataset. Finally write predictions to a JSON file."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Model, help="Model to use for prediction")
    parser.add_argument("--method", type=Methods, help="Method to use for prediction")
    parser.add_argument("--batch_size", type=int, help="Batch size for multi-field QA")
    args = parser.parse_args()

    # SQuAD dataset paths
    SQUAD_DEV_PATH = "data/asset/eval/squad_v2/dev-v2.0.json"
    SQUAD_DEV_TINY_PATH = "data/asset/eval/squad_v2/dev-v2.0_tiny.json"
    SQUAD_DEV_TEST_PATH = "data/asset/eval/squad_v2/dev-v2.0_test.json"
    SQUAD_DEV_TRAIN_PATH = "data/asset/eval/squad_v2/dev-v2.0_train.json"
    SQUAD_TRAIN_PATH = "data/asset/eval/squad_v2/train-v2.0.json"
    INPUT_PATH = (
        SQUAD_DEV_PATH  # @ Change this to the testing dataset path for experiments
    )
    OUTPUT_PRED_PATH = "data/asset/eval/squad_v2/results"

    OUTPUT_JSON_NAME = "{date}_pred_{model}_{method}_bs{bs}.json".format(
        date=datetime.datetime.now().strftime("%Y%m%d%H%M"),
        model=args.model.value,
        method=args.method.value,
        bs=args.batch_size,
    )

    # initialize the extractor
    model = OllamaLLM(model_name=args.model.value, temp=0.5, max_token=1024)
    squad_data_loader = SquadDatasetLoader()
    prompt_factory = PromptFactory()
    extractor = Extractor(
        model=model, data_loader=squad_data_loader, prompt_factory=prompt_factory
    )

    # run the batch operations
    if args.method == Methods.BATCH_QA:
        answers, summary, failure_log = extractor.batch_qa(
            batch_size=args.batch_size, dataset_path=INPUT_PATH
        )
    elif args.method == Methods.BATCH_STRUCTURED:
        answers, summary, failure_log = extractor.batch_structured(
            batch_size=args.batch_size, dataset_path=INPUT_PATH
        )
    elif args.method == Methods.BATCH_QA_STRUCTURED:
        answers, summary, failure_log = extractor.batch_qa_structured(
            batch_size=args.batch_size, dataset_path=INPUT_PATH
        )
    else:
        raise ValueError("Invalid method")

    # write results
    logger.info("Writing predictions to {}".format(OUTPUT_PRED_PATH))
    with open(os.path.join(OUTPUT_PRED_PATH, OUTPUT_JSON_NAME), "w") as f:
        json.dump(answers, f)

    logger.info("Writing summary to {}".format(OUTPUT_PRED_PATH))
    with open(
        os.path.join(
            OUTPUT_PRED_PATH, OUTPUT_JSON_NAME.replace(".json", "_summary.txt")
        ),
        "w",
    ) as f:
        f.write(json.dumps(summary, indent=4))

    if failure_log:
        logger.info("Writing failure log to {}".format(OUTPUT_PRED_PATH))
        with open(
            os.path.join(
                OUTPUT_PRED_PATH, OUTPUT_JSON_NAME.replace(".json", "_failure_log.txt")
            ),
            "w",
        ) as f:
            f.write(json.dumps(failure_log, indent=4))
    else:
        logger.info("No failures occurred. Skip writing failure log")
