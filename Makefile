ENV ?= gender_hw
ENV_FILE ?= environment.yml
CONFIG ?= configs/default.yaml
PY ?= conda run --no-capture-output -n $(ENV) python

.PHONY: setup train evaluate lint figures

setup:  ## новая conda-среда $(ENV) из $(ENV_FILE) + пакет; существующую среду не трогает
	@if conda env list | awk '{print $$1}' | grep -qx "$(ENV)"; then \
		echo "Среда '$(ENV)' уже существует — setup её не изменяет."; \
		echo "  обновить явно:   conda env update -n $(ENV) -f $(ENV_FILE) && conda run -n $(ENV) pip install -e '.[tabpfn,dev]'"; \
		echo "  или другое имя:  make setup ENV=<имя>"; \
		exit 1; \
	fi
	conda env create -n $(ENV) -f $(ENV_FILE)
	conda run -n $(ENV) pip install -e ".[tabpfn,dev]"

train:  ## данные, предобучение CoLES, кросс-валидация, финальные модели -> artifacts/, results/
	$(PY) -m gender_pred.cli train --config $(CONFIG)

evaluate:  ## единственный проход по test: метрики с 95% CI -> results/test_metrics.json
	$(PY) -m gender_pred.cli evaluate --config $(CONFIG)

lint:  ## линтер ruff (то же, что в CI)
	$(PY) -m ruff check src && $(PY) -m ruff format --check src

figures:  ## графики по results/ и data/train.csv -> reports/figures/
	$(PY) -m gender_pred.cli figures --config $(CONFIG)
