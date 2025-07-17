PYTHON_VERSION := 3.11

# Define virtualenv paths
TRAIN_VENV := .venv-train
EVAL_VENV  := .venv-eval

# Scripts
EVAL_SCRIPT := src/eval/eval.py
SFT_SCRIPT  := src/train/sft.py

# Configs (can be overridden from CLI)
EVAL_CONFIG ?= configs/eval/eval_config.yaml
SFT_CONFIG  ?= configs/train/sft_config.yaml

.PHONY: env-train env-eval eval sft clean

env-train:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up training environment..."
	@uv venv $(TRAIN_VENV) --python $(PYTHON_VERSION)
	@uv pip install --python $(TRAIN_VENV) -e ".[train]"
	@echo "Training environment ready."

env-eval:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up eval environment..."
	@uv venv $(EVAL_VENV) --python $(PYTHON_VERSION)
	@uv pip install --python $(EVAL_VENV) -e ".[eval]"
	@echo "Evaluation environment ready."

sft:
	@. $(TRAIN_VENV)/bin/activate && python $(SFT_SCRIPT) \
		--config $(SFT_CONFIG)

eval:
	@. $(EVAL_VENV)/bin/activate && python $(EVAL_SCRIPT) \
		--config $(EVAL_CONFIG)

clean:
	rm -rf $(TRAIN_VENV) $(EVAL_VENV)