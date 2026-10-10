# Пол клиента по карточным транзакциям

[![CI](https://github.com/Eltralo/gender-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Eltralo/gender-prediction/actions/workflows/ci.yml)

Бинарная классификация пола клиента банка по последовательности его транзакций (MCC, тип операции, сумма, время, терминал) на датасете [`mks-logic/gender_prediction`](https://huggingface.co/datasets/mks-logic/gender_prediction).

| | ROC AUC | Accuracy |
|---|---|---|
| Кросс-валидация на train + val, 2 × 5 фолдов | 0.8907 ± 0.0121 | 0.8044 ± 0.0132 |
| **Test**, 1 680 клиентов, 95% CI | **0.898** [0.883; 0.913] ✓ | **0.811** [0.793; 0.830] ✓ |
| Требование задания | > 0.88 | > 0.80 |
| Бейзлайн задания (RandomForest, 4 признака) | 0.574 | 0.558 |

```mermaid
flowchart LR
    D[Транзакции клиента] --> A[Агрегаты:<br/>MCC-профиль, суммы,<br/>время, динамика]
    D --> C[CoLES-энкодер<br/>GRU, предобучение без меток]
    C -->|эмбеддинг| A
    A --> G[LightGBM]
    A --> T[TabPFN]
    C --> F[CoLES, дообученный на пол]
    G --> B[Калибровка и бленд<br/>в пространстве логитов]
    T --> B
    F --> B
    B --> P[Вероятность и класс, порог 0.5]
```

```bash
make setup      # новая conda-среда gender_hw + пакет (без GPU: make setup ENV_FILE=environment-cpu.yml)
make evaluate   # метрики на test по финальной модели (артефакты из GitHub Release)
make train      # полный пересчёт: признаки, CoLES, кросс-валидация, финальные модели (~25 мин на GPU)
```

![Вклад компонентов](reports/figures/02_component_contribution.png)

![MCC с наибольшим перекосом по полу](reports/figures/01_mcc_gender_skew.png)

## Решение

1. **Агрегаты клиента** (~700 признаков без меток; вместе с признаками с метками — 719): активность, суммы расходов и поступлений, доли операций и трат по MCC и типам операций, средний чек по MCC, распределение по часам (время 00:00:00 трактуется как неизвестное) и дням недели, терминалы, концентрация трат, затухающие средние, автокорреляция, окна последних 30/90/180 дней. Словари категорий фиксируются по train, поэтому колонки одинаковы на любом сплите.
2. **Признаки с метками** — target encoding MCC и типов операций, логистическая регрессия по TF-IDF профилей и по эмбеддингу CoLES — считаются только внутри обучения: для обучающих клиентов out-of-fold по внутренним фолдам, для новых — по статистикам обучающих.
3. **CoLES**: GRU-энкодер (1024) над транзакциями предобучается контрастно без меток на клиентах train и val, затем даёт эмбеддинг-признак для бустинга и дообучается на пол.
4. **Модели и бленд**: LightGBM, дообученный CoLES и TabPFN v2 (300 признаков, отобранных внутри обучения). Скоры калибруются (Platt), бленд — взвешенная сумма логитов, веса подбираются по out-of-fold предсказаниям. Вероятности откалиброваны, поэтому порог 0.5 оптимален для accuracy без подбора.
5. **Валидация**: stratified 5-fold × 2 повтора на train + val; число итераций бустинга, калибровка и веса бленда для каждого фолда подбираются по остальным фолдам. Финальные модели обучаются на всех 6 720 размеченных клиентах. Test открыт один раз, на зафиксированной модели.

Подробности, вклад компонентов и проверенные, но не вошедшие идеи — в [REPORT.md](REPORT.md); рассказ с кодом и графиками — в [notebooks/pred_gen_tr.ipynb](notebooks/pred_gen_tr.ipynb).

## Запуск

**Быстрый старт** — ноутбук с готовыми моделями:

```bash
git clone https://github.com/Eltralo/gender-prediction.git && cd gender-prediction
conda env create -f environment.yml          # без GPU: environment-cpu.yml
conda activate gender_hw && pip install -e ".[tabpfn]"
jupyter lab notebooks/pred_gen_tr.ipynb      # Run → Run All Cells
```

Ноутбук сам скачает данные с Hugging Face и финальные модели из [GitHub Release](https://github.com/Eltralo/gender-prediction/releases/tag/v1.0.0) — авторизация не нужна. Финальная ячейка выведет **AUC 0.898 / accuracy 0.811** на test.

- `make evaluate` — то же без ноутбука: метрики на test с 95% CI → `results/test_metrics.json`.
- Обучение с нуля: `RETRAIN = True` в первой ячейке ноутбука или `make train` (~25 мин на GPU).
- `make lint` — проверка кода ruff (её же запускает CI); `make figures` — пересоздать графики в `reports/figures/`.

<details>
<summary>Подробнее: флаги пересчёта, артефакты, CPU, время</summary>

**Окружение.** `make setup` создаёт conda-среду из [environment.yml](environment.yml) (Python 3.11, PyTorch 2.11 с CUDA 12.8) и ставит пакет. Имя среды — переменная `ENV` (по умолчанию `gender_hw`), файл окружения — `ENV_FILE`: `make setup ENV=my_env ENV_FILE=environment-cpu.yml`. Существующую среду `setup` не изменяет: он остановится и подскажет команду для явного обновления. Цели `train` / `evaluate` / `lint` / `figures` запускаются в среде `ENV` через `conda run`.

**Данные** скачиваются с Hugging Face в `data/` автоматически (`python -m gender_pred.cli data`).

**Флаги пересчёта** — в [configs/default.yaml](configs/default.yaml):

| Флаг | По умолчанию | Действие |
|---|---|---|
| `retrain` | `false` | `false`: финальные модели из `artifacts/`; `true`: полный пересчёт |
| `retrain_tabular`, `retrain_seq` | `null` (= `retrain`) | пересчёт по частям: табличные модели и бленд / энкодер CoLES |

В ноутбуке тот же переключатель — `RETRAIN` в первой ячейке с кодом. Все гиперпараметры и сиды — в том же YAML.

**Артефакты.** Финальные модели (25 МБ) — архив `artifacts.zip` в GitHub Release v1.0.0. Источники перебираются по порядку:

1. уже распакованный `artifacts/` или архив, положенный вручную в `artifacts/artifacts.zip`;
2. публичная ссылка `artifacts_release.url` из конфига — без авторизации;
3. `gh release download` — если `gh` установлен и авторизован;
4. GitHub API с токеном из `GH_TOKEN` / `GITHUB_TOKEN`.

Если ни один источник недоступен, запуск останавливается с сообщением и вариантами действий; без артефактов модель можно обучить заново (`make train`).

**CPU.** Без GPU используйте `environment-cpu.yml` (PyTorch CPU-сборка); устройство выбирается автоматически (`device: auto`). Метрики выше получены на GPU; на CPU из-за численных различий GRU и TabPFN они могут отличаться в третьем-четвёртом знаке (проверено: AUC 0.8979, accuracy 0.8125 против 0.8113).

**Время** (RTX 5090 Laptop / 24 ядра CPU):

| | GPU | CPU |
|---|---|---|
| `make evaluate`, ноутбук с `RETRAIN = False` | ~2 мин | ~40 мин (эмбеддинги GRU и TabPFN на CPU) |
| `make train` (полный пересчёт) | ~25 мин | нецелесообразно: обучение GRU на CPU ≈ сутки |

</details>

## Структура

```
configs/default.yaml        гиперпараметры, сиды, флаги пересчёта
src/gender_pred/
    data.py                 загрузка, конфиг, подготовка транзакций, анализ данных
    features.py             агрегаты клиента и признаки с метками
    sequence.py             CoLES: энкодер, предобучение, дообучение
    models.py               LightGBM, TabPFN, калиброванный бленд
    cv.py                   кросс-валидация, финальное обучение, метрики, сила сигнала групп
    cli.py                  команды train / evaluate / figures
    plots.py                графики
notebooks/pred_gen_tr.ipynb ноутбук задания: анализ данных, признаки, модели, оценка
environment.yml             окружение (GPU); environment-cpu.yml — без CUDA
results/                    метрики CV и test, вклад компонентов, важности, анализ данных
reports/figures/            графики
```
