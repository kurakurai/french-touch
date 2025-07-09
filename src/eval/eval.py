import os
from pathlib import Path
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.models.utils import GenerationParameters
import argparse
import nltk


TASKS_REFS = {
    "ifeval-fr": "community|ifeval-fr|0|0",
    "gpqa-diamond-fr": "community|gpqa-diamond-fr|0|0",
    "bbh-fr": "community|bbh-fr|0|0",
    "boolq-fr": "community|boolq-fr|0|0",
    "mmlu-fr": "community|mmlu_fr|0|0",
    "musr-fr": "community|musr-fr|0|0",
    "math-hard-fr": "community|math-hard-fr|0|0",  # 4 shots under the hood
}


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

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=tasks_path,
    )

    generation_params = GenerationParameters(
        temperature=args.temperature,  # Set temperature to 0 for deterministic outputs
    )
    model_config = VLLMModelConfig(
        model_name=args.model,
        dtype="bfloat16",
        use_chat_template=True,
        generation_parameters=generation_params,
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
        choices=[
            "ifeval-fr",
            "gpqa-diamond-fr",
            "bbh-fr",
            "boolq-fr",
            "mmlu-fr",
            "musr-fr",
            "math-hard-fr",
        ],
        required=True,
        help="Tasks to evaluate the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_evals",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name to use for evaluation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation.",
    )
    args = parser.parse_args()
    main(args)
