PY ?= python
CONFIG ?= configs/default.yaml

.PHONY: setup train evaluate test figures

setup:  ## окружение conda + пакет в режиме разработки
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n gender_hw pip install -e ".[tabpfn,dev]"

train:  ## данные, предобучение CoLES, кросс-валидация, финальные модели -> artifacts/, results/
	$(PY) -m gender_pred.cli train --config $(CONFIG)

evaluate:  ## единственный проход по test: метрики с 95% CI -> results/test_metrics.json
	$(PY) -m gender_pred.cli evaluate --config $(CONFIG)

test:  ## линтер и быстрые тесты на синтетике (CPU)
	ruff check src tests && ruff format --check src tests
	CUDA_VISIBLE_DEVICES="" $(PY) -m pytest

figures:  ## графики по results/ -> reports/figures/
	$(PY) -m gender_pred.cli figures --config $(CONFIG)
