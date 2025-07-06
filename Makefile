PYTHON_VERSION       := 3.10
PYTHON               := uv run python

EVAL_SCRIPT		:= src/eval/eval.py

EVAL_TASKS ?= ifeval-fr
EVAL_MODEL ?= Qwen/Qwen3-0.6B

.PHONY: env eval

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up environment..."
	@uv sync --python $(PYTHON_VERSION)
	@echo "Environment ready."

eval:
	$(PYTHON) $(EVAL_SCRIPT) \
		--tasks $(EVAL_TASKS) \
		--model $(EVAL_MODEL) \