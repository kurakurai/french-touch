# French Small Language Model

Building a good French Small Language Model.

## 1. Quick Setup

_Using [`uv`](https://github.com/astral-sh/uv) for fast and reliable dependency management._

```bash
# Basic environment setup
make env
```
That's it, you can now run any command you want!

⚠️ You might need to perform the following two steps manually before running `make env`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## 2. Evaluation

Currently supports: `IFEval-fr`, `GPQA-Diamond-fr`, `BBH-fr`, `Math-HARD-fr`, `BoolQ-fr`, `MMLU-fr`, `MuSR-fr`

```bash
# Linux/MacOS
export HF_TOKEN=your_hf_token
# Windows
$env:HF_TOKEN="your_hf_token"
```

| Task        | Make Command       | Equivalent CLI Command                                                                                                                                               | Default Values                                                                 |
|-------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Evaluation French Benchmarks   | `make eval`       | `python src/eval/eval.py --tasks EVAL_TASKS --model EVAL_MODEL`                                                                                 | `EVAL_TASKS=ifeval-fr`, `EVAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct`                              |

⚠️ We use [Lighteval](https://github.com/huggingface/lighteval) and [vLLM](https://github.com/vllm-project/vllm) for evaluation.

## 3. Results

| Evaluation               | Qwen3-0.6B   | Qwen2.5-0.5B-Instruct |
|--------------------------|--------------|-----------------------|
| IFEval-fr (strict prompt)|              |  18.43                |
| GPQA-Diamond-fr          |              |  35.84                |
| BoolQ-fr                 |              |  0.00                 |
| Math-Hard-fr             |              |  1.40                 |
| MMLU-fr                  |              |  35.37                |
| BBH-fr                   |              |  42.33                |
| MuSR-fr                  |              |  37.49                |

| Evaluation               | Qwen2.5-1.5B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B | Qwen3-1.7B |
|--------------------------|-----------------------|-------------------------------|------------|
| IFEval-fr (strict prompt)|    26.63              |                               |            |
| GPQA-Diamond-fr          |    30.07              |                               |            |
| BoolQ-fr                 |    70.39              |                               |            |
| Math-Hard-fr             |    4.64               |                               |            |
| MMLU-fr                  |    48,17              |                               |            |
| BBH-fr                   |    47.57              |                               |            |
| MuSR-fr                  |    37.47              |                               |            |

## 3. Resources:
- [French LLM Leaderboard](https://huggingface.co/spaces/fr-gouv-coordination-ia/llm_leaderboard_fr#/)
- [OpenLLMFrenchLeadboard Dataset (not official datasets)](https://huggingface.co/collections/le-leadboard/openllmfrenchleadboard-jeu-de-donnees-67126437539a23c65554fd88)
