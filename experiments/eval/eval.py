import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

PROJECT_ROOT = "/home/yixin0829/minmod/minmod-poc/"


def run_squad_evals(path_to_dev: str, path_to_predictions: str):
    # Run the squad.py script with the given arguments
    result = subprocess.run(
        [
            "python",
            PROJECT_ROOT + "experiments/eval/squad.py",
            PROJECT_ROOT + path_to_dev,
            PROJECT_ROOT + path_to_predictions,
        ],
        capture_output=True,
        text=True,
    )

    # Capture the output and return it
    return result.stdout


def plot_bar_chart(experiment_results: dict[str, dict[str, float]]):
    # Metrics to plot
    metrics_to_plot = [
        "exact",
        "f1",
        "HasAns_exact",
        "HasAns_f1",
        "NoAns_exact",
        "NoAns_f1",
    ]

    # Create a DataFrame from the experiment results with three columns: experiment, metric, and value
    df = pd.DataFrame(
        [
            {"experiment": experiment, "metric": metric, "value": value}
            for experiment, metrics in experiment_results.items()
            for metric, value in metrics.items()
            if metric in metrics_to_plot
        ]
    )

    # Creating a bar plot using Seaborn with figure size 12x6
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="metric", y="value", hue="experiment")
    plt.show()


def main():
    args_combinations = [
        (
            "QA BS=2",
            "data/asset/eval/squad_v2/dev-v2.0.json",
            "data/asset/eval/squad_v2/results/202406082346_pred_llama3-custom_batch-qa_bs2.json",
        ),
        (
            "QA BS=4",
            "data/asset/eval/squad_v2/dev-v2.0.json",
            "data/asset/eval/squad_v2/results/202406090123_pred_llama3-custom_batch-qa_bs4.json",
        ),
        (
            "QA BS=8",
            "data/asset/eval/squad_v2/dev-v2.0.json",
            "data/asset/eval/squad_v2/results/202406090243_pred_llama3-custom_batch-qa_bs8.json",
        ),
        (
            "JSON BS=2",
            "data/asset/eval/squad_v2/dev-v2.0.json",
            "data/asset/eval/squad_v2/results/202406090357_pred_llama3-custom_batch-structured_bs2.json",
        ),
        (
            "JSON BS=4",
            "data/asset/eval/squad_v2/dev-v2.0.json",
            "data/asset/eval/squad_v2/results/202406082311_pred_llama3-custom_batch-structured_bs4.json",
        ),
        (
            "JSON BS=8",
            "data/asset/eval/squad_v2/dev-v2.0.json",
            "data/asset/eval/squad_v2/results/202406082325_pred_llama3-custom_batch-structured_bs8.json",
        ),
    ]

    results = {}

    # Run squad.py with different argument combinations
    for exp_name, path_to_dev, path_to_predictions in args_combinations:
        output = run_squad_evals(path_to_dev, path_to_predictions)

        # Assume the output is in JSON format for this example
        try:
            output_dict = json.loads(output)
            results[exp_name] = output_dict
        except json.JSONDecodeError:
            print(
                f"Failed to parse JSON output for {path_to_dev} and {path_to_predictions}"
            )

    # plot the results
    plot_bar_chart(results)


if __name__ == "__main__":
    main()
