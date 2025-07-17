PYTHON_VERSION       := 3.11
PYTHON               := uv run python

EVAL_SCRIPT          := src/eval/eval.py
SFT_SCRIPT           := src/train/sft.py

EVAL_CONFIG          ?= configs/eval/eval_config.yaml
SFT_CONFIG           ?= configs/train/sft_config.yaml

.PHONY: env eval sft

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

sft: env
	$(PYTHON) $(SFT_SCRIPT) \
		--config $(SFT_CONFIG) \
