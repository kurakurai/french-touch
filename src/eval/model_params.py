from lighteval.models.utils import GenerationParameters

MODEL_PARAMS = {
    "Qwen/Qwen3-0.6B": GenerationParameters(  # Non-thinking mode
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
    ),
    "Qwen/Qwen3-1.7B": GenerationParameters(  # Non-thinking mode
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
    ),
    "Qwen/Qwen2.5-0.5B-Instruct": GenerationParameters(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1,
        presence_penalty=0,
        frequency_penalty=0,
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": GenerationParameters(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1,
        presence_penalty=0,
        frequency_penalty=0,
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": GenerationParameters(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    ),
    "LiquidAI/LFM2-1.2B": GenerationParameters(
        temperature=0.3, min_p=0.15, repetition_penalty=1.05
    ),
    "LiquidAI/LFM2-700M": GenerationParameters(
        temperature=0.3, min_p=0.15, repetition_penalty=1.05
    ),
}
