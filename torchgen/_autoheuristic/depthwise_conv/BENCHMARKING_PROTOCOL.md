# Depthwise Conv Heuristic Benchmarking Protocol

This protocol is for collecting depthwise convolution heuristic datasets on
different GPU machine types, then training and validating the generated
decision trees.

## Per-Machine Setup

Run these commands from `torchgen/_autoheuristic/depthwise_conv` on the branch
that contains the current benchmarking scripts:

```
python gen_data_depthwiseconv.py --help
python train_decision_depthwiseconv.py --help
```

The benchmark requires CUDA and cuDNN. It writes one CSV and one
`*.metadata.json` sidecar for each run. Keep the CSV and metadata sidecar
together when copying results between machines.

Use a short, stable device label in every command. Examples:

|GPU|Suggested label|
|---|---|
|T4|`T4`|
|A10|`A10`|
|A100|`A100`|
|L40S|`L40S`|
|H100|`H100`|
|GB200|`GB200`|
|RTX5090|`RTX5090`|

## Collect Data

For a new machine, collect all three grids:

```
python gen_data_depthwiseconv.py \
  --grid train \
  --device DEVICE \
  --output data_depthwiseconv_train_DEVICE.csv

python gen_data_depthwiseconv.py \
  --grid validation \
  --device DEVICE \
  --output data_depthwiseconv_validation_DEVICE.csv

python gen_data_depthwiseconv.py \
  --grid test \
  --device DEVICE \
  --output data_depthwiseconv_test_DEVICE.csv
```

Expected row counts:

|Grid|Rows|Purpose|
|---|---:|---|
|`train`|1584|Used to train the decision tree for that SM group|
|`validation`|594|Used for tuning features and tree complexity|
|`test`|288|Used only for final reporting|

If the base train grid was already collected before `train-extra` existed,
collect only the incremental train rows:

```
python gen_data_depthwiseconv.py \
  --grid train-extra \
  --device DEVICE \
  --output data_depthwiseconv_train_extra_DEVICE.csv
```

The `train-extra` grid has 576 rows. It should be combined with the older
1008-row train CSV when training.

Use `--benchmark-iters` and `--warmup-iters` only when intentionally running a
short smoke test. The default iteration counts should be used for production
datasets unless all machines use the same override.

## Check Collected Files

Run this quick check after each collection:

```
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

for csv in Path(".").glob("data_depthwiseconv_*_DEVICE.csv"):
    df = pd.read_csv(csv)
    metadata = json.loads(csv.with_suffix(".metadata.json").read_text())
    print(csv)
    print("  rows:", len(df))
    print("  sm:", sorted(df["sm"].unique()))
    print("  grid:", metadata["benchmark_grid"]["name"])
    print("  workload_count:", metadata["benchmark_grid"]["workload_count"])
    print("  device:", metadata["device"]["name"])
    print("  cudnn_frontend:", metadata["cudnn_frontend_version"])
    print("  cudnn_backend:", metadata["cudnn_backend_version"])
PY
```

Replace `DEVICE` with the label used in the output filenames. The row count
should match the grid. The metadata should identify the expected GPU, PyTorch
commit, cuDNN frontend version, cuDNN backend version, collection date, and SM.

## SM Grouping

Training builds one tree per cuDNN SM heuristic group:

|cuDNN SM group|Examples|
|---|---|
|`sm75`|T4, RTX8000|
|`sm80`|A100|
|`sm86`|A10|
|`sm89`|L40S, L40, L40G, RTX4090|
|`sm90`|H100, H200, GH100, GH200|
|`sm100`|B200, GB200, B300, GB300, Thor|
|`sm120`|RTX5080, RTX5090, PRO 6000 Blackwell, DGX Spark|

The `sm` column is used only to select the SM group. It is not used as a
decision-tree feature. Runtime dispatch uses SM ranges, so missing groups fall
back to the nearest generated tree.

## Train and Validate

After copying datasets from all machines to one checkout, train with all train
CSVs and hold out validation/test CSVs:

```
python train_decision_depthwiseconv.py \
  data_depthwiseconv_train_*.csv \
  --min-samples-leaf 16 \
  --validation-files data_depthwiseconv_validation_*.csv \
  --test-files data_depthwiseconv_test_*.csv
```

The current `train` grid already includes the targeted extra cases, so a single
set of `train` CSVs is sufficient. If you have older datasets that were split
into a separate `train-extra` CSV, add `data_depthwiseconv_train_extra_*.csv` to
the input file list as well.

This regenerates:

```
../../../aten/src/ATen/autoheuristic/DepthwiseConvHeuristic.h
```

To compare experiments without overwriting the checked-in header, pass
`--output-file`:

```
python train_decision_depthwiseconv.py \
  data_depthwiseconv_train_*.csv \
  --min-samples-leaf 16 \
  --validation-files data_depthwiseconv_validation_*.csv \
  --test-files data_depthwiseconv_test_*.csv \
  --output-file /tmp/DepthwiseConvHeuristic_experiment.h
```

The trainer prints:

- tree size per generated SM group
- train accuracy per generated SM group
- validation accuracy and confusion matrix
- test accuracy and confusion matrix
- aggregate policy metrics versus always-cuDNN and oracle

Use validation results to tune feature choices or complexity knobs. Use test
results only for the final report.

## Smoke Tests

Run the reduced SymInt and benchmark-path check after script or heuristic
changes:

```
python check_symint_depthwiseconv.py --benchmark-iters 1 --warmup-iters 1
```

Run syntax and whitespace checks before sharing results:

```
python -m py_compile \
  gen_data_depthwiseconv.py \
  train_decision_depthwiseconv.py \
  check_symint_depthwiseconv.py

git diff --check
```

## Files to Share Back

For each machine, share:

- `data_depthwiseconv_train_DEVICE.csv`
- `data_depthwiseconv_train_DEVICE.metadata.json`
- `data_depthwiseconv_validation_DEVICE.csv`
- `data_depthwiseconv_validation_DEVICE.metadata.json`
- `data_depthwiseconv_test_DEVICE.csv`
- `data_depthwiseconv_test_DEVICE.metadata.json`

If only the incremental train grid was collected, share:

- `data_depthwiseconv_train_extra_DEVICE.csv`
- `data_depthwiseconv_train_extra_DEVICE.metadata.json`

Do not rename files after collection unless the metadata sidecar stays paired
with the matching CSV.
