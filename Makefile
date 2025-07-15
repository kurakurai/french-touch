PYTHON_VERSION       := 3.10
PYTHON               := uv run python

EVAL_SCRIPT          := src/eval/eval.py

EVAL_CONFIG          ?= configs/eval_config.yaml

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
		--config $(EVAL_CONFIG) \

