# Forest test fixtures

Small XGBoost and LightGBM models plus fixed inputs and expected outputs, for
`algeon-cuml`'s tree-model loading and prediction tests. Every model is
trained on synthetic data (or handcrafted directly) with a fixed numpy seed,
so regeneration is deterministic and reproducible byte-for-byte.

## Regeneration

```bash
source ~/.venvs/forest/bin/activate
CUDA_VISIBLE_DEVICES="" python3 \
    components/cuml/algeon-cuml/tests/fixtures/forest/generate.py
```

The script sets `CUDA_VISIBLE_DEVICES=""` itself before importing xgboost or
lightgbm, and never imports `nvforest`, so it never touches a GPU. It writes
every file next to itself (resolved from `__file__`), so it can be run from
any working directory. Re-running it reproduces byte-identical output; this
was verified by running it twice into separate directories and diffing.

## Versions used

See `versions.txt`, written by the script on every run. At the time these
fixtures were generated:

```
python=3.12.3
xgboost=3.4.1
lightgbm=4.7.0
treelite=4.7.2
numpy=2.5.3
```

## File formats

- `<name>.input.csv`: one row per line, comma-separated float32 values, no
  header. Each value is the shortest decimal string that round-trips to the
  exact float32 bit pattern
  (`np.format_float_positional(np.float32(v), unique=True, trim='-')`), so
  Rust's `str::parse::<f32>()` reproduces the identical bits.
- `<name>.expected.csv`: one row per input row, comma-separated outputs of
  the source framework's own predictor (`xgboost.Booster.predict` /
  `lightgbm.Booster.predict`) on the float32 input, written with
  `repr(float(x))` (full double precision).
- `<name>.gtil.csv`: same shape as `.expected.csv`, output of
  `treelite.gtil.predict(model, X)` where `model` is loaded by treelite from
  the same model file (`treelite.frontend.load_xgboost_model` /
  `load_lightgbm_model`). GTIL's raw return shape is `(rows, 1, k)`; the
  script flattens it to `(rows,)` or `(rows, k)` before writing.
- `manifest.csv`: header `name,model_file,format,rows,features,expected_columns,notes`,
  one row per fixture. `format` is `xgboost_json` or `lightgbm`. `notes` uses
  semicolons instead of commas (no quoting), so the file stays splittable by
  a plain `line.split(',')`.
- `versions.txt`: `key=value` lines, one per package, written by the script.

For fixtures 1-6 the script asserts `expected` and `gtil` agree (elementwise,
within `1e-5` for continuous outputs, or via matching argmax for class-index
outputs) and prints the max deviation; all six pass at generation time.

## Fixtures

1. **`xgb_binary_logistic`** (`xgb_binary_logistic.json`, XGBoost JSON).
   `binary:logistic`, 200 synthetic training rows, 4 trees, depth 3. 16 input
   rows, 4 features. `expected`/`gtil`: 1 column, predicted probability.

2. **`xgb_multiclass_softmax`** (`xgb_multiclass_softmax.json`, XGBoost
   JSON). `multi:softmax`, `num_class=3`, same training setup (4 trees,
   depth 3). 16 input rows, 4 features. `expected` (`Booster.predict`): 1
   column, predicted class index as a float (e.g. `2.0`).
   `gtil`: **3 columns**, not 1. Treelite assigns this model
   `postprocessor=softmax` (verified via
   `treelite.frontend.load_xgboost_model(...).dump_as_json()["postprocessor"]`
   on treelite 4.7.2), not `max_index`; GTIL therefore returns per-class
   softmax probabilities, the same shape as `xgb_multiclass_softprob`'s
   GTIL output. `argmax` of each `gtil.csv` row reproduces the class index
   in `expected.csv` exactly; the generator asserts this instead of a
   direct elementwise comparison.

3. **`xgb_multiclass_softprob`** (`xgb_multiclass_softprob.json`, XGBoost
   JSON). `multi:softprob`, `num_class=3`, same training data and
   hyperparameters as fixture 2. 16 input rows, 4 features. `expected`/
   `gtil`: 3 probability columns per row.

4. **`xgb_multiclass_softprob_ubj`** (`xgb_multiclass_softprob_ubj.ubj`,
   XGBoost UBJSON). The exact same trained `Booster` object as fixture 3
   (`xgb_multiclass_softprob`), saved with `Booster.save_model(...)` to a
   `.ubj` path instead of `.json`; XGBoost picks the UBJSON writer from the
   extension. No separate input/expected/gtil CSVs: this row reuses
   `xgb_multiclass_softprob.input.csv`, `.expected.csv`, and `.gtil.csv`
   verbatim (recorded in its `manifest.csv` notes), since the model and data
   are identical and only the on-disk encoding differs. The generator still
   reloads the `.ubj` file through both `xgboost.Booster.load_model` and
   `treelite.frontend.load_xgboost_model` and checks predictions match
   fixture 3's within `1e-5`, to confirm the UBJSON round-trip is faithful.

5. **`lgb_regression_double`** (`lgb_regression_double.txt`, LightGBM text
   model). `regression` objective, 200 training rows with `float64` `X` and
   `y` (so split thresholds in the saved text model are doubles), 4 trees
   (`num_boost_round=4`), `num_leaves=8`, `max_depth=3`. 16 input rows, 4
   features (float32, per the general input format; the model's own
   thresholds are the doubles from training). `expected`/`gtil`: 1 column.

6. **`xgb_multiclass_tie`** (`xgb_multiclass_tie.json`, XGBoost JSON,
   handcrafted). `multi:softmax`, `num_class=3`. Not trained: the JSON is
   built directly as 3 leaf-only (no-split) trees, one per class:
   class 0 leaf value `-1.0`, classes 1 and 2 leaf value `0.5`. `base_score`
   is `0` for every class, so the raw per-class score is exactly the leaf
   value on every row, with classes 1 and 2 tied above class 0 on all 16
   rows. `xgboost.Booster.predict` returns class `1` (the lower of the tied
   indices) on every row; this is asserted in the generator.
   `gtil` (3 columns, `postprocessor=softmax` as in fixture 2): `argmax` of
   each row also picks class 1, matching `expected.csv`.

7. **`lgb_threshold_boundary`** (`lgb_threshold_boundary.txt`, LightGBM
   text model, handcrafted). One tree, one split on feature 0,
   `decision_type=2` (numerical `<=`, default-left), threshold
   `0.99999998509883881` (= `1 - 2**-26`, written with 17 significant
   digits), left leaf (the `<=` branch) `-1`, right leaf (the `>` branch)
   `+1`. Built by training a throwaway single-split 1-feature model to get a
   structurally valid LightGBM v4 text file, then editing only the
   `threshold=` and `leaf_value=` lines in place; every other bookkeeping
   field is left as LightGBM wrote it. `np.float32(threshold) == 1.0` is
   asserted (a float32 cast of the threshold would flip the branch decision
   for input `1.0`), and `1.0 > threshold` in float64, so the model and
   GTIL both predict `+1.0` for input `1.0`; this is asserted for both. 4
   input rows, 1 feature: `1.0`, `0.99999994` (the float32 value just below
   `1.0`), `0.5`, `2.0`. `expected`/`gtil`: 1 column,
   `[1.0, -1.0, -1.0, 1.0]`.

8. **`xgb_multi_target`** (`xgb_multi_target.json`, XGBoost JSON).
   `reg:squarederror` with a 2-column target (`multi_strategy` left at its
   default, `one_output_per_tree`), 200 training rows, 4 trees, depth 3. 16
   input rows, 4 features. No `expected.csv`/`gtil.csv`: this fixture only
   exercises the multi-target rejection path. The generator asserts
   `treelite.frontend.load_xgboost_model(...).dump_as_json()["num_target"] == 2`.

9. **`corrupt.json`**: the first half of `xgb_binary_logistic.json`'s bytes
   (byte-truncated, not JSON-truncated), for JSON parse-error tests. Not
   valid JSON by construction (confirmed: `json.load` raises
   `JSONDecodeError`).

## Determinism

Every random draw goes through a `numpy.random.default_rng(seed)` with a
fixed integer seed local to that fixture (101/102 for fixture 1, 201/202 for
fixtures 2-3 and fixture 4's shared `X_in`, 301/302 for fixture 5, 401 for
fixture 6, 601 for fixture 7's throwaway pretraining, 501/502 for fixture 8).
No global numpy seed or process-wide RNG state is used, so fixtures do not
depend on generation order.

## A postprocessor note on `max_index`

An earlier revision of this generator planned two additional fixtures built
directly with `treelite.model_builder.ModelBuilder`, with `postprocessor`
set to `"max_index"`, to exercise a lowest-index-on-tie argmax path
independent of XGBoost's own softmax mapping. Treelite 4.7.2 cannot
round-trip such a model. The legacy-checkpoint deserializer in
`src/serializer.cc` rewrites `postprocessor == "max_index"` to `"softmax"`
on load (`model->postprocessor = "softmax"; // max_index no longer supported`),
and GTIL's `postprocessor.cc` has no `"max_index"` entry in
`GetPostProcessorFunc`. A freshly built `"max_index"` model's default
(non-`pred_margin`) `treelite.gtil.predict` call therefore raises
`TreeliteError: Post-processor named 'max_index' not found`. No
`tl_max_index` / `tl_max_index_tie` fixtures exist here for that reason.
`xgb_multiclass_tie` (fixture 6) covers the lowest-index tie through the
`softmax` postprocessor that XGBoost `multi:softmax` models receive.
