#!/usr/bin/env python3
"""Generate small tree-model test fixtures for algeon-cuml's forest tests.

Deterministic: every random draw uses a fixed-seed numpy Generator. Writes all
output files next to this script (resolved from __file__, independent of the
caller's cwd) so it can be run from any directory.

Regenerate with:

    source ~/.venvs/forest/bin/activate
    CUDA_VISIBLE_DEVICES="" python3 \
        components/cuml/algeon-cuml/tests/fixtures/forest/generate.py

GPU is never used: CUDA_VISIBLE_DEVICES is forced empty before xgboost and
lightgbm are imported, and no nvforest import occurs anywhere in this script.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
import lightgbm as lgb
import treelite
import treelite.gtil as gtil

HERE = Path(__file__).resolve().parent

# XGBoost's default training-history callback evaluates the training metric
# (e.g. merror for multiclass) after every boosting round, even with no
# `evals` passed. On this machine that eval path probes for a CUDA driver
# and raises cudaErrorNoDevice; it does not fall back to CPU, so every
# xgboost.train() call here disables the default eval metric.
XGB_BASE_PARAMS = {"device": "cpu", "verbosity": 0, "disable_default_eval_metric": 1}


def fmt_f32(v):
    """Shortest decimal string that round-trips to this exact float32 value."""
    return np.format_float_positional(np.float32(v), unique=True, trim="-")


def fmt_f64(v):
    return repr(float(v))


def write_csv(path, rows, fmt):
    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(fmt(v) for v in np.atleast_1d(row)) + "\n")


def write_input(path, X):
    """X: float32 ndarray, shape (rows, features)."""
    write_csv(path, X, fmt_f32)


def write_output(path, Y):
    """Y: ndarray (any float dtype), shape (rows, cols) or (rows,)."""
    Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
    write_csv(path, Y, fmt_f64)


def assert_close(name, expected, actual, rtol=1e-5):
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.shape != actual.shape:
        raise AssertionError(f"{name}: shape mismatch {expected.shape} vs {actual.shape}")
    diff = np.abs(expected - actual)
    worst = float(diff.max())
    if worst > rtol:
        raise AssertionError(f"{name}: max abs diff {worst} exceeds {rtol}")
    print(f"  [ok] {name}: max abs diff {worst:.3e} (<= {rtol})")


def assert_exact(name, expected, actual):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        raise AssertionError(f"{name}: shape mismatch {expected.shape} vs {actual.shape}")
    if not np.array_equal(expected, actual):
        bad = np.flatnonzero(expected != actual)
        raise AssertionError(f"{name}: mismatch at rows {bad.tolist()}")
    print(f"  [ok] {name}: exact match ({expected.size} values)")


results = []  # (name, path, size_bytes) collected for the final report


def record(path):
    results.append((path.name, path.stat().st_size))


# Fixture 1: xgb_binary_logistic


def build_xgb_binary_logistic():
    name = "xgb_binary_logistic"
    rng = np.random.default_rng(101)
    X = rng.uniform(-3.0, 3.0, size=(200, 4)).astype(np.float32)
    logit = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(200) < p).astype(np.int32)

    params = dict(XGB_BASE_PARAMS, objective="binary:logistic", max_depth=3, eta=0.3)
    bst = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=4)

    model_path = HERE / f"{name}.json"
    bst.save_model(str(model_path))
    record(model_path)

    rng_in = np.random.default_rng(102)
    X_in = rng_in.uniform(-3.0, 3.0, size=(16, 4)).astype(np.float32)
    input_path = HERE / f"{name}.input.csv"
    write_input(input_path, X_in)
    record(input_path)

    expected = bst.predict(xgb.DMatrix(X_in))
    expected_path = HERE / f"{name}.expected.csv"
    write_output(expected_path, expected)
    record(expected_path)

    model = treelite.frontend.load_xgboost_model(str(model_path))
    g = gtil.predict(model, X_in)
    assert g.shape == (16, 1, 1), f"unexpected gtil shape {g.shape}"
    g = g.reshape(16)
    gtil_path = HERE / f"{name}.gtil.csv"
    write_output(gtil_path, g)
    record(gtil_path)

    print(f"{name}:")
    assert_close(name, expected, g)
    return name, model_path.name, "xgboost_json", 16, 4, 1, "binary:logistic probability, 1 column"


# Fixtures 2 & 3: xgb_multiclass_softmax / xgb_multiclass_softprob


def _multiclass_data():
    rng = np.random.default_rng(201)
    X = rng.uniform(-3.0, 3.0, size=(200, 4)).astype(np.float32)
    noise = rng.normal(scale=0.3, size=(200, 3))
    scores = np.stack(
        [X[:, 0] + X[:, 1], X[:, 2] - X[:, 3], -X[:, 0] + X[:, 3]], axis=1
    ) + noise
    y = np.argmax(scores, axis=1).astype(np.int32)

    rng_in = np.random.default_rng(202)
    X_in = rng_in.uniform(-3.0, 3.0, size=(16, 4)).astype(np.float32)
    return X, y, X_in


def build_xgb_multiclass_softmax(X, y, X_in):
    name = "xgb_multiclass_softmax"
    params = dict(XGB_BASE_PARAMS, objective="multi:softmax", num_class=3, max_depth=3, eta=0.3)
    bst = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=4)

    model_path = HERE / f"{name}.json"
    bst.save_model(str(model_path))
    record(model_path)

    input_path = HERE / f"{name}.input.csv"
    write_input(input_path, X_in)
    record(input_path)

    expected = bst.predict(xgb.DMatrix(X_in))  # class index, float-valued
    expected_path = HERE / f"{name}.expected.csv"
    write_output(expected_path, expected)
    record(expected_path)

    model = treelite.frontend.load_xgboost_model(str(model_path))
    g = gtil.predict(model, X_in)
    assert g.shape == (16, 1, 3), f"unexpected gtil shape {g.shape}"
    g = g.reshape(16, 3)
    gtil_path = HERE / f"{name}.gtil.csv"
    write_output(gtil_path, g)
    record(gtil_path)

    print(f"{name}:")
    assert_exact(name, expected.astype(np.int64), np.argmax(g, axis=1).astype(np.int64))
    notes = (
        "expected = xgboost class index (1 column); treelite assigns this "
        "XGBoost multi:softmax model postprocessor=softmax (not max_index), "
        "so gtil.csv holds 3 softmax probability columns, and "
        "argmax(gtil row) reproduces the expected class index"
    )
    return name, model_path.name, "xgboost_json", 16, 4, 1, notes


def build_xgb_multiclass_softprob(X, y, X_in):
    name = "xgb_multiclass_softprob"
    params = dict(XGB_BASE_PARAMS, objective="multi:softprob", num_class=3, max_depth=3, eta=0.3)
    bst = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=4)

    model_path = HERE / f"{name}.json"
    bst.save_model(str(model_path))
    record(model_path)

    input_path = HERE / f"{name}.input.csv"
    write_input(input_path, X_in)
    record(input_path)

    expected = bst.predict(xgb.DMatrix(X_in))  # (16, 3) probabilities
    expected_path = HERE / f"{name}.expected.csv"
    write_output(expected_path, expected)
    record(expected_path)

    model = treelite.frontend.load_xgboost_model(str(model_path))
    g = gtil.predict(model, X_in)
    assert g.shape == (16, 1, 3), f"unexpected gtil shape {g.shape}"
    g = g.reshape(16, 3)
    gtil_path = HERE / f"{name}.gtil.csv"
    write_output(gtil_path, g)
    record(gtil_path)

    print(f"{name}:")
    assert_close(name, expected, g)
    manifest_row = (name, model_path.name, "xgboost_json", 16, 4, 3, "3 class-probability columns per row")
    return manifest_row, bst


def build_xgb_multiclass_softprob_ubj(bst, X_in):
    """Same trained booster as xgb_multiclass_softprob, saved as XGBoost UBJSON.

    Reuses xgb_multiclass_softprob.input.csv / .expected.csv / .gtil.csv: no
    new CSVs are written, only the .ubj model file, since the model and data
    are identical, only the on-disk encoding differs.
    """
    name = "xgb_multiclass_softprob_ubj"
    model_path = HERE / f"{name}.ubj"
    bst.save_model(str(model_path))
    record(model_path)

    expected = bst.predict(xgb.DMatrix(X_in))
    model = treelite.frontend.load_xgboost_model(str(model_path))
    g = gtil.predict(model, X_in)
    assert g.shape == (16, 1, 3), f"unexpected gtil shape {g.shape}"
    g = g.reshape(16, 3)

    print(f"{name}:")
    assert_close(name, expected, g)
    notes = (
        "same trained booster as xgb_multiclass_softprob, saved as XGBoost "
        "UBJSON instead of JSON; reuses "
        "xgb_multiclass_softprob.input.csv/.expected.csv/.gtil.csv verbatim, "
        "no separate CSVs for this row"
    )
    return name, model_path.name, "xgboost_ubjson", 16, 4, 3, notes


# Fixture 4: lgb_regression_double


def build_lgb_regression_double():
    name = "lgb_regression_double"
    rng = np.random.default_rng(301)
    X = rng.uniform(-3.0, 3.0, size=(200, 4)).astype(np.float64)
    y = (2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3]) + rng.normal(scale=0.1, size=200)
    y = y.astype(np.float64)

    ds = lgb.Dataset(X, label=y)
    params = {
        "objective": "regression",
        "num_leaves": 8,
        "max_depth": 3,
        "min_data_in_leaf": 5,
        "verbosity": -1,
    }
    bst = lgb.train(params, ds, num_boost_round=4)

    model_path = HERE / f"{name}.txt"
    bst.save_model(str(model_path))
    record(model_path)

    rng_in = np.random.default_rng(302)
    X_in = rng_in.uniform(-3.0, 3.0, size=(16, 4)).astype(np.float32)
    input_path = HERE / f"{name}.input.csv"
    write_input(input_path, X_in)
    record(input_path)

    expected = bst.predict(X_in)
    expected_path = HERE / f"{name}.expected.csv"
    write_output(expected_path, expected)
    record(expected_path)

    model = treelite.frontend.load_lightgbm_model(str(model_path))
    g = gtil.predict(model, X_in)
    assert g.shape == (16, 1, 1), f"unexpected gtil shape {g.shape}"
    g = g.reshape(16)
    gtil_path = HERE / f"{name}.gtil.csv"
    write_output(gtil_path, g)
    record(gtil_path)

    print(f"{name}:")
    assert_close(name, expected, g)
    return name, model_path.name, "lightgbm", 16, 4, 1, "regression output, 1 column; model trained on float64 data (double thresholds)"


# Fixture 5: xgb_multiclass_tie (handcrafted)


def _leaf_only_tree(value, tree_id):
    return {
        "base_weights": [value],
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
        "default_left": [0],
        "id": tree_id,
        "left_children": [-1],
        "right_children": [-1],
        "loss_changes": [0.0],
        "parents": [2147483647],
        "split_conditions": [value],
        "split_indices": [0],
        "split_type": [0],
        "sum_hessian": [1.0],
        "tree_param": {
            "num_deleted": "0",
            "num_feature": "4",
            "num_nodes": "1",
            "size_leaf_vector": "1",
        },
    }


def build_xgb_multiclass_tie():
    name = "xgb_multiclass_tie"
    # One stump (leaf-only, no split) per class: class 0 always scores -1.0,
    # classes 1 and 2 always score 0.5, an exact tie above class 0 on every
    # row. base_score is 0 for every class so the raw score is the leaf
    # value with no additive offset to break the tie.
    trees = [_leaf_only_tree(-1.0, 0), _leaf_only_tree(0.5, 1), _leaf_only_tree(0.5, 2)]
    model_json = {
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "model": {
                    "cats": {"enc": [], "feature_segments": [], "sorted_idx": []},
                    "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": "3"},
                    "iteration_indptr": [0, 3],
                    "tree_info": [0, 1, 2],
                    "trees": trees,
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": "[0E0,0E0,0E0]",
                "boost_from_average": "0",
                "num_class": "3",
                "num_feature": "4",
                "num_target": "1",
            },
            "objective": {"name": "multi:softmax", "softmax_multiclass_param": {"num_class": "3"}},
        },
        "version": [3, 4, 1],
    }

    model_path = HERE / f"{name}.json"
    with open(model_path, "w") as f:
        json.dump(model_json, f, indent=1)
    record(model_path)

    bst = xgb.Booster()
    bst.load_model(str(model_path))

    rng_in = np.random.default_rng(401)
    X_in = rng_in.uniform(-3.0, 3.0, size=(16, 4)).astype(np.float32)
    input_path = HERE / f"{name}.input.csv"
    write_input(input_path, X_in)
    record(input_path)

    expected = bst.predict(xgb.DMatrix(X_in))
    assert np.all(expected == 1.0), f"expected every row to pick class 1 on tie, got {expected}"
    expected_path = HERE / f"{name}.expected.csv"
    write_output(expected_path, expected)
    record(expected_path)

    model = treelite.frontend.load_xgboost_model(str(model_path))
    g = gtil.predict(model, X_in)
    assert g.shape == (16, 1, 3), f"unexpected gtil shape {g.shape}"
    g = g.reshape(16, 3)
    gtil_path = HERE / f"{name}.gtil.csv"
    write_output(gtil_path, g)
    record(gtil_path)

    print(f"{name}:")
    assert_exact(name, expected.astype(np.int64), np.argmax(g, axis=1).astype(np.int64))
    notes = (
        "handcrafted: 3 leaf-only (no-split) trees, class 0 leaf -1.0, "
        "classes 1 and 2 leaf 0.5 on every row, base_score 0 for all "
        "classes; xgboost predict picks the lowest tied index (class 1) on "
        "every row. gtil.csv holds 3 softmax probability columns (same "
        "postprocessor note as xgb_multiclass_softmax); argmax matches."
    )
    return name, model_path.name, "xgboost_json", 16, 4, 1, notes


# Fixture 6: lgb_threshold_boundary (handcrafted)


def build_lgb_threshold_boundary():
    name = "lgb_threshold_boundary"
    threshold = 1.0 - 2.0**-26
    assert np.float32(threshold) == np.float32(1.0), "threshold must round up to 1.0f in float32"
    threshold_str = "%.17g" % threshold
    assert threshold_str == "0.99999998509883881", threshold_str

    # Train a throwaway single-split, 1-feature model purely to obtain a
    # structurally valid LightGBM v4 text model to edit in place: writing
    # the whole text format from scratch would have to reproduce every
    # bookkeeping field LightGBM's parser expects.
    rng = np.random.default_rng(601)
    X = rng.uniform(-2.0, 2.0, size=(200, 1))
    y = (X[:, 0] > 0.0).astype(np.float64) * 2.0 - 1.0
    ds = lgb.Dataset(X, label=y)
    params = {
        "objective": "regression",
        "num_leaves": 2,
        "max_depth": 1,
        "min_data_in_leaf": 1,
        "verbosity": -1,
    }
    bst = lgb.train(params, ds, num_boost_round=1)
    scratch_path = HERE / f"{name}._scratch.txt"
    bst.save_model(str(scratch_path))
    text = scratch_path.read_text()
    scratch_path.unlink()

    # decision_type=2 (numerical, "<=" split, default_left) and the
    # left_child=-1/right_child=-2 leaf layout are unchanged from the
    # trained template; only the threshold and the two leaf values are
    # handcrafted. left leaf (index 0) is the "<=" branch, right leaf
    # (index 1) is the ">" branch.
    old_threshold_line = next(l for l in text.splitlines() if l.startswith("threshold="))
    old_leaf_line = next(l for l in text.splitlines() if l.startswith("leaf_value="))
    text = text.replace(old_threshold_line, f"threshold={threshold_str}")
    text = text.replace(old_leaf_line, "leaf_value=-1 1")

    model_path = HERE / f"{name}.txt"
    model_path.write_text(text)
    record(model_path)

    bst_edited = lgb.Booster(model_file=str(model_path))
    X_in = np.array([[1.0], [0.99999994], [0.5], [2.0]], dtype=np.float32)
    input_path = HERE / f"{name}.input.csv"
    write_input(input_path, X_in)
    record(input_path)

    expected = bst_edited.predict(X_in)
    assert expected[0] == 1.0, f"input 1.0 must predict +1.0 (1.0 > threshold), got {expected[0]}"
    expected_path = HERE / f"{name}.expected.csv"
    write_output(expected_path, expected)
    record(expected_path)

    model = treelite.frontend.load_lightgbm_model(str(model_path))
    g = gtil.predict(model, X_in)
    assert g.shape == (4, 1, 1), f"unexpected gtil shape {g.shape}"
    g = g.reshape(4)
    assert g[0] == 1.0, f"gtil must also predict +1.0 for input 1.0, got {g[0]}"
    gtil_path = HERE / f"{name}.gtil.csv"
    write_output(gtil_path, g)
    record(gtil_path)

    print(f"{name}:")
    assert_close(name, expected, g)
    notes = (
        "handcrafted single-split tree, 1 feature, threshold = 1 - 2**-26 "
        "= 0.99999998509883881 written with 17 significant digits; "
        "np.float32(threshold) == 1.0, so a float32 cast of the threshold "
        "flips the branch decision for input 1.0. Rows: 1.0, 0.99999994 "
        "(float32 just below 1), 0.5, 2.0"
    )
    return name, model_path.name, "lightgbm", 4, 1, 1, notes


# Fixture 7: xgb_multi_target (rejection fixture, no expected/gtil)


def build_xgb_multi_target():
    name = "xgb_multi_target"
    rng = np.random.default_rng(501)
    X = rng.uniform(-3.0, 3.0, size=(200, 4)).astype(np.float32)
    Y = np.stack([2.0 * X[:, 0] - X[:, 1], X[:, 2] + 0.5 * X[:, 3]], axis=1).astype(np.float32)

    params = dict(XGB_BASE_PARAMS, objective="reg:squarederror", max_depth=3, eta=0.3)
    bst = xgb.train(params, xgb.DMatrix(X, label=Y), num_boost_round=4)

    model_path = HERE / f"{name}.json"
    bst.save_model(str(model_path))
    record(model_path)

    rng_in = np.random.default_rng(502)
    X_in = rng_in.uniform(-3.0, 3.0, size=(16, 4)).astype(np.float32)
    input_path = HERE / f"{name}.input.csv"
    write_input(input_path, X_in)
    record(input_path)

    model = treelite.frontend.load_xgboost_model(str(model_path))
    d = json.loads(model.dump_as_json())
    assert d["num_target"] == 2, f"expected num_target == 2, got {d['num_target']}"

    print(f"{name}: [ok] treelite num_target == 2 (no expected/gtil files; rejection fixture)")
    return name, model_path.name, "xgboost_json", 16, 4, 0, "multi-target model (num_target=2); rejection-path fixture, no expected/gtil files"


# Fixture 8: corrupt.json (truncated copy of fixture 1)


def build_corrupt(binary_logistic_model_path):
    name = "corrupt"
    data = binary_logistic_model_path.read_bytes()
    truncated = data[: len(data) // 2]
    model_path = HERE / f"{name}.json"
    model_path.write_bytes(truncated)
    record(model_path)
    print(f"{name}: truncated {len(data)} bytes -> {len(truncated)} bytes")
    return name, model_path.name, "xgboost_json", 0, 0, 0, "first half of xgb_binary_logistic.json bytes, for parse-error tests"


def write_versions():
    versions = {
        "python": platform.python_version(),
        "xgboost": importlib.metadata.version("xgboost"),
        "lightgbm": importlib.metadata.version("lightgbm"),
        "treelite": importlib.metadata.version("treelite"),
        "numpy": importlib.metadata.version("numpy"),
    }
    path = HERE / "versions.txt"
    with open(path, "w") as f:
        for k, v in versions.items():
            f.write(f"{k}={v}\n")
    record(path)
    return versions


def write_manifest(rows):
    # No quoting: notes use semicolons, and commas only separate fields, so the file
    # stays splittable by a naive `line.split(',')`, matching the other fixture
    # text formats (no CSV/JSON library on the Rust side).
    path = HERE / "manifest.csv"
    with open(path, "w") as f:
        f.write("name,model_file,format,rows,features,expected_columns,notes\n")
        for name, model_file, fmt, n_rows, n_features, n_cols, notes in rows:
            notes = notes.replace(",", ";").replace("\n", " ")
            f.write(f"{name},{model_file},{fmt},{n_rows},{n_features},{n_cols},{notes}\n")
    record(path)


def main():
    if sys.platform != "linux":
        raise RuntimeError("this generator targets Linux only, per repository policy")

    manifest_rows = []

    r1 = build_xgb_binary_logistic()
    manifest_rows.append(r1)
    model1_path = HERE / f"{r1[0]}.json"

    X_mc, y_mc, X_mc_in = _multiclass_data()
    manifest_rows.append(build_xgb_multiclass_softmax(X_mc, y_mc, X_mc_in))
    softprob_row, softprob_bst = build_xgb_multiclass_softprob(X_mc, y_mc, X_mc_in)
    manifest_rows.append(softprob_row)
    manifest_rows.append(build_xgb_multiclass_softprob_ubj(softprob_bst, X_mc_in))

    manifest_rows.append(build_lgb_regression_double())
    manifest_rows.append(build_xgb_multiclass_tie())
    manifest_rows.append(build_lgb_threshold_boundary())
    manifest_rows.append(build_xgb_multi_target())
    manifest_rows.append(build_corrupt(model1_path))

    versions = write_versions()
    write_manifest(manifest_rows)

    print("\nversions:")
    for k, v in versions.items():
        print(f"  {k}={v}")

    print("\nfiles written:")
    for filename, size in sorted(results):
        print(f"  {filename}: {size} bytes")


if __name__ == "__main__":
    main()
