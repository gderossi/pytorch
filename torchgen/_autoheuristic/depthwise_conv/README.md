## Regenerating the current heuristic
Data collection is a multi-machine process. See `BENCHMARKING_PROTOCOL.md` for
how to collect a train/validation/test dataset on each GPU type and what to
share back.

Once the per-machine CSVs (and their `*.metadata.json` sidecars) have been
gathered into one checkout, regenerate the checked-in heuristic with:

```
python train_decision_depthwiseconv.py \
  <train CSVs ...> \
  --min-samples-leaf 16 \
  --validation-files <validation CSVs ...> \
  --test-files <test CSVs ...>
```

`--min-samples-leaf 16` is the setting used for the checked-in trees; it keeps
the generated trees from overfitting to individual benchmark points. The
validation and test files are never used for training; they only produce the
holdout accuracy, confusion-matrix, and policy-metric reports.

## Dataset provenance
The benchmark grid and data-generation procedure are defined in
`gen_data_depthwiseconv.py`. Every run writes a `*.metadata.json` sidecar next
to its CSV recording the PyTorch commit, device name/type, SM, cuDNN frontend
version, cuDNN backend version, collection date, and benchmark grid.

A separate decision tree is generated per cuDNN SM heuristic group. The
checked-in heuristic was built from the following machines. Each machine
contributed a train (1584 rows), validation (594 rows), and test (288 rows)
split. Train/val/test accuracy is the per-group accuracy reported by the
training run that produced the current header.

|SM group tree|Machine(s)|Device name(s)|SM|Train acc|Val acc|Test acc|
|---|---|---|---|---:|---:|---:|
|`sm80`|A100|NVIDIA A100-PCIE-40GB|80|93.75%|89.73%|88.89%|
|`sm89`|L40|NVIDIA L40|89|90.33%|86.68%|88.50%|
|`sm90`|H100|NVIDIA H100 80GB HBM3|90|93.04%|92.42%|88.50%|
|`sm100`|GB200, GB300, Thor|NVIDIA GB200 / GB300 / Thor|100 / 103 / 110|87.68%|86.59%|85.68%|
|`sm120`|RTX PRO 6000, Spark|NVIDIA RTX PRO 6000 Blackwell Server Edition / GB10|120 / 121|89.96%|87.24%|87.54%|

Aggregate holdout numbers across all groups: overall validation accuracy
87.89%, overall test accuracy 87.25%. On the test set the generated heuristic
reaches a 1.935x mean speedup over the native kernel, versus 1.876x for
always-cuDNN and 1.963x for an oracle that picks the faster kernel per shape
(the heuristic is within ~1.5% of the oracle).

All datasets were collected at PyTorch commit
`5140876473ecc91ca14c695d8a40690f4c8919f3` with cuDNN frontend version 1.22.1
(`12201`) and cuDNN backend version 9.23.0 (`92300`), on 2026-06-12 through
2026-06-15.

The `sm75` (T4, RTX8000) and `sm86` (A10) groups are not generated because that
hardware was not available when this dataset was collected. Runtime dispatch
falls back to the nearest generated tree for those devices (`sm75` -> `sm80`,
`sm86` -> `sm89`); see the SM grouping notes below.

## Heuristic generation
To generate a new heuristic from benchmarking data, run the training script:

```
python train_decision_depthwiseconv.py \
  [train input files ...] \
  --validation-files [validation files ...] \
  --test-files [test files ...] \
  --output-file OUTPUT_FILE
```

At least one input file must be provided. If multiple files are provided, the training script will first combine all inputs into
one dataset.
Validation and test files are not used to train the decision trees. They are
used only to print holdout accuracy, confusion matrices, and aggregate policy
metrics after the heuristic is generated. The policy metrics compare the
generated heuristic against always using cuDNN and against an oracle that picks
the faster measured implementation for each row.

The generated trees use `bs`, `ch`, `h`, `w`, `filter`, `stride`, `out_h`,
`out_w`, `out_elements`, and `kernel_work` as decision variables. Older
square-only CSVs without an `h` column are still supported; the training script
treats them as `h == w`.

The training script attempts to generate a separate decision tree for each
cuDNN SM heuristic group: `sm75`, `sm80`, `sm86`, `sm89`, `sm90`, `sm100`,
and `sm120`. The `sm` column is used only to assign samples to one of these
groups; it is not used as a feature inside the decision trees. Runtime dispatch
uses SM ranges so devices from new architecture revisions fall back to an
existing generated tree instead of requiring an exact SM match. If a group has
no training samples, it falls back to the nearest generated group.

There are several options that can be provided to the training script, with `--tolerance` being the most important. This option
excludes data from decision tree training if the normalized speedup value is within tolerance, which can make a large impact on
final tree complexity and accuracy. For example, consider the following benchmarking results with `--tolerance 0.1`:

|id|cudnn_speedup_all|
|---|---|
|A|1.4|
|B|0.95|
|C|1.05|
|D|0.8|

Samples A and D will be included in training, because they are outside the tolerance window of `0.1`:
```
abs(1.4 - 1) > 0.1
abs(0.8 - 1) > 0.1
```
Samples B and C will be excluded from training, because they are within the tolerance window of `0.1`.

### Full options information:
```
usage: train_decision_depthwiseconv.py [-h] [--tolerance TOLERANCE] [--max-depth MAX_DEPTH] [--max-leaf-nodes MAX_LEAF_NODES]
                                       [--min-samples-split MIN_SAMPLES_SPLIT] [--min-samples-leaf MIN_SAMPLES_LEAF]
                                       [--criterion {gini,entropy,log_loss}] [--seed SEED]
                                       [--output-file OUTPUT_FILE]
                                       [--validation-files VALIDATION_FILES [VALIDATION_FILES ...]]
                                       [--test-files TEST_FILES [TEST_FILES ...]]
                                       input_files [input_files ...]

positional arguments:
  input_files           Paths to processed CSV files

options:
  -h, --help            show this help message and exit
  --tolerance TOLERANCE
                        Tolerance threshold (default: 0.0)
  --max-depth MAX_DEPTH
                        Maximum tree depth (default: None = unlimited)
  --max-leaf-nodes MAX_LEAF_NODES
                        Maximum number of leaf nodes (default: None = unlimited)
  --min-samples-split MIN_SAMPLES_SPLIT
                        Minimum samples to split a node (default: 2)
  --min-samples-leaf MIN_SAMPLES_LEAF
                        Minimum samples in leaf node (default: 1)
  --criterion {gini,entropy,log_loss}
                        Split criterion (default: gini)
  --seed SEED           Random seed (default: 42)
  --output-file OUTPUT_FILE
                        Output header path
  --validation-files VALIDATION_FILES [VALIDATION_FILES ...]
                        Optional holdout CSV files used only for validation
  --test-files TEST_FILES [TEST_FILES ...]
                        Optional final holdout CSV files used only for reporting
```
