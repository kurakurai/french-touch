import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from patch_lighteval.patch import patch_reasoning

patch_reasoning()
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.models.utils import GenerationParameters
from pathlib import Path
from utils import MODEL_PARAMS, TASKS_REFS
import argparse
import nltk
import numpy as np


def get_tasks(task_keys):
    """
    Get the tasks based on the provided task keys.
    """
    tasks = []
    for key in task_keys:
        if key in TASKS_REFS:
            tasks.append(TASKS_REFS[key])
        else:
            raise ValueError(f"Task {key} is not defined in TASKS_REFS.")
    return ",".join(tasks)


def display_avg_metrics(all_results):
    """Calculate and display average metrics across multiple runs."""
    other_keys = [k for k in all_results[0].keys() if k != "all"]

    print(f"\nAVERAGE RESULTS ACROSS {len(all_results)} RUNS:")
    print("=" * 50)

    for section in other_keys:
        print(f"\n>>> {section.upper()}")
        print("-" * 50)
        section_keys = all_results[0][section].keys()
        base_keys = sorted([k for k in section_keys if not k.endswith("_stderr")])

        for key in base_keys:
            values = [result[section][key] for result in all_results]
            stderrs = [result[section][f"{key}_stderr"] for result in all_results]
            mean = np.mean(values)
            avg_stderr = np.mean(stderrs)
            print(f"{key}: {mean:.4f} ± {avg_stderr:.4f} = {mean + avg_stderr:.4f}")


def main(args):
    """
    Main function to run the evaluation pipeline with vLLM backend.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt_tab")

    tasks_path = Path(f"src/eval/tasks.py")

    config_kwargs = {
        "model_name": args.model,
        "dtype": "bfloat16",
        "use_chat_template": True,
    }

    if args.model in MODEL_PARAMS:
        config_kwargs["generation_parameters"] = GenerationParameters(
            **MODEL_PARAMS[args.model]
        )

    model_config = VLLMModelConfig(**config_kwargs)

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=tasks_path,
        use_chat_template=True,  # Set false for base models
    )
    tasks = get_tasks(args.tasks)
    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
        push_to_hub=False,
    )

    all_results = []
    for _ in range(args.num_runs):
        pipeline = Pipeline(
            tasks=tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
            enable_thinking=args.enable_thinking,  # Enable or disable reasoning (default is False)
        )
        pipeline.evaluate()
        all_results.append(
            pipeline.evaluation_tracker.metrics_logger.metric_aggregated.copy()
        )
        pipeline.save_and_push_results()
        pipeline.show_results()

    # Calculate average metrics across all runs
    if args.num_runs > 1:
        display_avg_metrics(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with vLLM backend.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[
            "ifeval-fr",
            "gpqa-diamond-fr",
            "bbh-fr",
            "boolq-fr",
            "mmlu-fr",
            "musr-fr",
            "math-hard-fr",
            "hellaswag-fr",
        ],
        required=True,
        help="Tasks to evaluate the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name to use for evaluation.",
    )

    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable reasoning mode for the model.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=2,
        help="Number of times to run each task.",
    )
    args = parser.parse_args()
    main(args)
