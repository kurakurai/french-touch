TASKS_REFS = {
    "ifeval-fr": "community|ifeval-fr|0|0",
    "gpqa-diamond-fr": "community|gpqa-diamond-fr|0|0",
    "bbh-fr": "community|bbh-fr|0|0",
    "boolq-fr": "community|boolq-fr|0|0",
    "mmlu-fr": "community|mmlu_fr|0|0",
    "musr-fr": "community|musr-fr|0|0",
    "math-hard-fr": "community|math-hard-fr|0|0",  # 4 shots under the hood
}

MODEL_PARAMS = {
    "Qwen/Qwen3-0.6B": {  # Non-thinking mode
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    },
    "Qwen/Qwen3-1.7B": {  # Non-thinking mode
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    },
    "LiquidAI/LFM2-1.2B": {
        "temperature": 0.3,
        "min_p": 0.15,
        "repetition_penalty": 1.05,
    },
    "LiquidAI/LFM2-700M": {
        "temperature": 0.3,
        "min_p": 0.15,
        "repetition_penalty": 1.05,
    },
}
