## Regenerating the current heuristic
For the full multi-machine collection workflow, see
`BENCHMARKING_PROTOCOL.md`.

To regenerate the current heuristic with the downloaded data plus any newly
collected train, validation, and test splits, run:

```
bash get_depthwiseconv_dataset.sh

python gen_data_depthwiseconv.py --grid train --device DEVICE
python gen_data_depthwiseconv.py --grid validation --device DEVICE
python gen_data_depthwiseconv.py --grid test --device DEVICE

python train_decision_depthwiseconv.py \
  [train input files ...] \
  --validation-files [validation files ...] \
  --test-files [test files ...]
```

## Dataset provenance
The benchmark grid and data-generation procedure are defined in
`gen_data_depthwiseconv.py`. The benchmark script writes a
`*.metadata.json` sidecar next to each generated CSV with the PyTorch commit,
device type/name, SM, cuDNN frontend version, cuDNN backend version, collection
date, and benchmark grid.

The checked-in heuristic was generated from these training inputs:

|Input|Rows|SM group|
|---|---|---|
|`data_depthwiseconv_A100.csv`|2016|sm80|
|`data_depthwiseconv_H100.csv`|2016|sm90|
|`data_depthwiseconv_GB200.csv`|2016|sm100|
|`data_depthwiseconv_GB300.csv`|2016|sm100|
|`data_depthwiseconv_protocol_train_local.csv`|1008|sm100|
|`data_depthwiseconv_protocol_train_extra_local.csv`|576|sm100|

The local protocol data and holdout results were collected with:

|Field|Value|
|---|---|
|PyTorch commit|f4bdea026bdf3e86cf13216a83d6672b29ebc69e|
|Device|NVIDIA GB200 (CUDA, sm100)|
|cuDNN frontend version|12201 (1.22.1)|
|cuDNN backend version|92300 (9.23.0)|
|Collection date|2026-06-12|

The local validation holdout has 594 rows and the local final test holdout has
288 rows.

## Benchmarking protocol
Collect train, validation, and final test data separately:

```
python gen_data_depthwiseconv.py --grid train --device DEVICE
python gen_data_depthwiseconv.py --grid validation --device DEVICE
python gen_data_depthwiseconv.py --grid test --device DEVICE
```

The `--device` flag is optional, and allows you to specify a custom shorter
label for the GPU being tested. Output will be saved to
`data_depthwiseconv_[GRID]_[DEVICE].csv` unless `--output` is provided. Use
`--benchmark-iters` and `--warmup-iters` to override the default per-shape
iteration counts.

The train grid includes the original square shape family plus additional
non-square shapes, plus targeted square cases with `bs={4,16,64}` and
`ch={64,128,256,512}` to cover gaps from the original downloaded grid. The
`train-extra` grid can be used to benchmark only those targeted additional
cases when the base train grid has already been collected. The validation grid
uses different batch sizes, channels, and shapes from the train grid and is
used while tuning tree features or model complexity. The test grid is smaller,
fully held out, and should be used for final reporting only.

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
