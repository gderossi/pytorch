## Regenerating the current heuristic
To regenerate the current heuristic with the original data, run the following scripts:

```
bash get_depthwiseconv_dataset.sh

python train_decision_depthwiseconv.py *.csv
```

## Dataset provenance
The benchmark grid and data-generation procedure are defined in
`gen_data_depthwiseconv.py`. The benchmark script writes a
`*.metadata.json` sidecar next to each generated CSV with the PyTorch commit,
device type/name, SM, cuDNN frontend version, cuDNN backend version, collection
date, and benchmark grid.

The checked-in heuristic was generated from data collected with:

|Field|Value|
|---|---|
|PyTorch commit|655c28ee371d46906d6af0697c36718089edc173|
|Device|NVIDIA GB200 (CUDA, sm100)|
|cuDNN frontend version|12201 (1.22.1)|
|cuDNN backend version|92300 (9.23.0)|
|Collection date|2026-06-12|

## Benchmarking
To collect new data, run the benchmarking script:

`python gen_data_depthwiseconv.py [--device DEVICE] [--grid {train,validation}] [--output OUTPUT]`

The `--device` flag is optional, and allows you to specify a custom (shorter) label for the GPU being tested.
Output will be saved to `data_depthwiseconv_[GRID]_[DEVICE].csv` unless `--output` is provided. Depending on the GPU, the train grid will likely take 30 minutes to 2 hours.
The validation grid uses shapes that are not in the train grid and is intended
to be benchmarked separately as a holdout set.

## Heuristic generation
To generate a new heuristic from benchmarking data, run the training script:

`python train_decision_depthwiseconv.py [input files ...] [--validation-files VALIDATION_FILES ...] [--output-file OUTPUT_FILE]`

At least one input file must be provided. If multiple files are provided, the training script will first combine all inputs into
one dataset.
Validation files are not used to train the decision trees; they are used only
to print holdout accuracy and confusion matrices after the heuristic is
generated.

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
```
