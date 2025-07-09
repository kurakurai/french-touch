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
| Evaluation French Benchmarks   | `make eval`       | `python run_eval.py --tasks EVAL_TASKS --model EVAL_MODEL`                                                                                 | `EVAL_TASKS=ifeval_fr`, `EVAL_MODEL=Qwen/Qwen3-0.6B`                              |

⚠️ We use [Lighteval](https://github.com/huggingface/lighteval) and [vLLM](https://github.com/vllm-project/vllm) for evaluation.

## 3. Results

Parameters: `Temperature=0.0`

| Evaluation               | Qwen2.5-1.5B | 
|--------------------------|--------------|
| IFEval-fr (strict prompt)| 23.78        |
| GPQA-Diamond-fr          | 27.44        |
| BoolQ-fr                 | 69.3         | 
| Math-Hard-fr             | 4.77         | 
| MMLU-fr                  | 48.17        | 
| BBH-fr                   | -            | 
| MuSR-fr                  |              | 


## 3. Resources:
- [French LLM Leaderboard](https://huggingface.co/spaces/fr-gouv-coordination-ia/llm_leaderboard_fr#/)
- [OpenLLMFrenchLeadboard Dataset (not official datasets)](https://huggingface.co/collections/le-leadboard/openllmfrenchleadboard-jeu-de-donnees-67126437539a23c65554fd88)
