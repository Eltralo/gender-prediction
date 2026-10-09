import numpy as np
import pandas as pd

from gender_pred import cli, features
from gender_pred.data import Vocab, labels, prepare_events


def test_columns_match_across_splits(splits, tiny_cfg):
    vocab = Vocab.from_train(splits["train"])
    cols = [
        features.client_features(prepare_events(df), vocab, tiny_cfg["features"]).columns
        for df in splits.values()
    ]
    assert all(list(c) == list(cols[0]) for c in cols)


def test_end_to_end_shapes_and_ranges(splits, tiny_cfg):
    model = cli.train(tiny_cfg, splits=splits)
    m = cli.evaluate(tiny_cfg, model=model, splits=splits)
    pred = pd.read_parquet(f"{tiny_cfg['paths']['results']}/test_predictions.parquet")
    assert len(pred) == splits["test"]["customer_id"].nunique()
    assert pred["proba"].between(0, 1).all()
    assert 0 <= m["auc_ci"][0] <= m["auc"] <= m["auc_ci"][1] <= 1
    assert set(model.oof_.index) == set(labels(pd.concat([splits["train"], splits["val"]])).index)


def test_test_labels_not_used(splits, tiny_cfg):
    """Перестановка меток test не меняет ни OOF, ни веса бленда, ни предсказания."""
    a = cli.train(tiny_cfg, splits=splits)
    shuffled = dict(splits)
    test = splits["test"].copy()
    flip = dict(
        zip(
            test["customer_id"].unique(),
            np.random.default_rng(0).permutation(test.groupby("customer_id")["gender"].first().to_numpy()),
        )
    )
    test["gender"] = test["customer_id"].map(flip)
    shuffled["test"] = test
    b = cli.train(tiny_cfg, splits=shuffled)
    pd.testing.assert_frame_equal(a.oof_, b.oof_)
    np.testing.assert_allclose(a.blend_.weights_, b.blend_.weights_)
