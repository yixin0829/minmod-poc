import json
from dataclasses import dataclass

from squad import compute_exact, compute_f1, normalize_answer


@dataclass
class ErrorAnalysisArgs:
    experiment_name: str
    path_to_ref: str
    path_to_qa_preds: str
    path_to_structured_preds: str


args_combinations = [
    (
        "BS=2",
        "data/asset/eval/squad_v2/dev-v2.0.json",
        "data/asset/eval/squad_v2/results/202406082346_pred_llama3-custom_batch-qa_bs2.json",
        "data/asset/eval/squad_v2/results/202406090357_pred_llama3-custom_batch-structured_bs2.json",
    ),
    (
        "BS=4",
        "data/asset/eval/squad_v2/dev-v2.0.json",
        "data/asset/eval/squad_v2/results/202406090123_pred_llama3-custom_batch-qa_bs4.json",
        "data/asset/eval/squad_v2/results/202406111210_pred_llama3-custom_batch-structured_bs4.json",
    ),
    (
        "BS=8",
        "data/asset/eval/squad_v2/dev-v2.0.json",
        "data/asset/eval/squad_v2/results/202406090243_pred_llama3-custom_batch-qa_bs8.json",
        "data/asset/eval/squad_v2/results/202406111755_pred_llama3-custom_batch-structured_bs8.json",
    ),
]

# convert to ErrorAnalysisArgs
error_analysis_args = [ErrorAnalysisArgs(*args) for args in args_combinations]

# loop through error_analysis_args
for args in error_analysis_args:
    print("Running error analysis for experiment: {}".format(args.experiment_name))
    # read JSON from paths
    with open(args.path_to_ref) as f:
        data_json = json.load(f)
        dataset = data_json["data"]
    with open(args.path_to_qa_preds) as f:
        qa_preds = json.load(f)
    with open(args.path_to_structured_preds) as f:
        structured_preds = json.load(f)

    error_analysis = {}
    for i, article in enumerate(dataset):
        for j, paragraph in enumerate(article["paragraphs"]):
            for k, qa in enumerate(paragraph["qas"]):
                qa_id = qa["id"]
                gold_answers = [
                    a["text"] for a in qa["answers"] if normalize_answer(a["text"])
                ]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = [""]
                qa_pred = qa_preds[qa_id]
                structured_pred = structured_preds[qa_id]

                exact_scores_qa = max(compute_exact(a, qa_pred) for a in gold_answers)
                exact_scores_structured = max(
                    compute_exact(a, structured_pred) for a in gold_answers
                )

                if exact_scores_qa and not exact_scores_structured:
                    error_analysis[qa_id] = {
                        "question_id": qa_id,
                        "is_impossible": qa["is_impossible"],
                        "question": qa["question"],
                        "context": paragraph["context"],
                        "gold_answers": gold_answers,
                        "qa_pred": qa_pred,
                        "structured_pred": structured_pred,
                        "exact_scores_qa": exact_scores_qa,
                        "exact_scores_structured": exact_scores_structured,
                    }

    # write error_analysis to JSON
    with open(
        "data/asset/eval/squad_v2/results/error_analysis_{}.json".format(
            args.experiment_name
        ),
        "w",
    ) as f:
        json.dump(error_analysis, f, indent=4)
