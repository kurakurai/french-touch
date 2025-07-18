# Luth: Small French Language Model

Building a Small French Language Model.

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

Custom French evals supported: `IFEval-fr`, `GPQA-Diamond-fr`, `BBH-fr`, `Math-HARD-fr`, `BoolQ-fr`, `MMLU-fr`, `MuSR-fr`, `Hellaswag-fr`

```bash
# Linux/MacOS
export HF_TOKEN=your_hf_token
# Windows
$env:HF_TOKEN="your_hf_token"
```

| Task        | Make Command       | Equivalent CLI Command                                                                                                                                               | Default Values                                                                 |
|-------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Evaluation French Benchmarks   | `make eval`       | `python src/eval/eval.py --config EVAL_CONFIG`                                                                                 | `EVAL_CONFIG=configs/eval_config.yaml`                     |

⚠️ We use [LightEval](https://github.com/huggingface/lighteval) and [vLLM](https://github.com/vllm-project/vllm) for evaluation.

## 3. Results

### 3.1 Non-Thinking mode

| Benchmark                 | Qwen3-0.6B | Qwen2.5-0.5B-Instruct | SmolLM2-360M-Instruct |  LFM2-700M   |  LFM2-350M   |
|---------------------------|------------|-----------------------|-----------------------|--------------|--------------|
| IFEval-fr (prompt strict) |  **43.44** | 20.70                 | 20.89                 |              |              |
| GPQA-Diamond-fr           |  **29.95** | 22.84                 | 28.43                 |              |              |
| BBH-fr                    |    24.60   | 24.35                 |  **40.41**            |              |              |
| Math-HARD-fr              |  **10.86** | 3.04                  |   1.17                |              |              |
| BoolQ-fr                  |  **74.72** | 59.55                 |   26.97               |              |              |
| MMLU-fr                   |  27.16     | **35.04**             |   26.14               |              |              |
| MuSR-fr                   |  33.34     |   34.17               |   **36.41**           |              |              |
| Hellaswag-fr              |  24.90     | **51.21**             |   32.78               |              |              |

We used `temperature=0.0` and `system_prompt="Vous êtes un assistant utile."`.