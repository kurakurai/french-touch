import os
from pathlib import Path
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
import argparse
import nltk

# https://github.com/leaderboard-modeles-IA-francais/evaluation-pipeline-leaderboard/blob/main/run-lighteval.py


def get_tasks(tasks):
    task_lookup = {
        "ifeval_fr": "community|ifeval-fr|0|0",
        "gpqa_fr": "community|gpqa-fr|0|0",
    }
    selected_tasks = ",".join([task_lookup[t] for t in tasks])
    return selected_tasks


def main(args):
    """
    Main function to run the evaluation pipeline with vLLM backend.
    """

    if os.environ.get("HF_TOKEN") is None:
        raise ValueError(
            "Please set the HF_TOKEN environment variable to your Hugging Face token."
        )

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt_tab")

    tasks_path = Path(f"src/eval/french_evals.py")

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=tasks_path,
    )

    # Can add more parameters here
    model_config = VLLMModelConfig(
        model_name=args.model,
        dtype="bfloat16",
        use_chat_template=True,
    )

    pipeline = Pipeline(
        tasks=get_tasks(args.tasks),
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with vLLM backend.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["ifeval_fr", "gpqa_fr"],
        default="ifeval_fr",
        help="Tasks to evaluate the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_french_evals",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name to use for evaluation.",
    )
    args = parser.parse_args()
    main(args)
