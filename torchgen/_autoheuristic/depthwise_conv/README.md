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
|`sm75`|T4|Tesla T4|75|96.39%|93.60%|89.58%|
|`sm80`|A100|NVIDIA A100-PCIE-40GB|80|93.75%|89.73%|88.89%|
|`sm86`|A10|NVIDIA A10|86|96.47%|93.43%|90.29%|
|`sm89`|L40|NVIDIA L40|89|90.33%|86.68%|88.50%|
|`sm90`|H100|NVIDIA H100 80GB HBM3|90|93.04%|92.42%|88.50%|
|`sm100`|GB200, GB300, Thor|NVIDIA GB200 / GB300 / Thor|100 / 103 / 110|87.68%|86.59%|85.68%|
|`sm120`|RTX PRO 6000, Spark|NVIDIA RTX PRO 6000 Blackwell Server Edition / GB10|120 / 121|89.96%|87.24%|87.54%|

Aggregate holdout numbers across all groups: overall validation accuracy
89.01%, overall test accuracy 87.78%. On the test set, the regenerated
heuristic disagrees with the legacy `check_cudnn_depthwise_workload` heuristic
from `Convolution.cpp` on 530 rows. Excluding rows where both heuristics make
the same decision, the regenerated heuristic is 1.159x faster than the legacy
heuristic by total runtime. On just those disagreement rows, the legacy
heuristic reaches a 0.916x speedup over the native kernel, the regenerated
heuristic reaches 1.062x, and the oracle reaches 1.068x.

|Device|Disagreement rows|New vs legacy speedup|Legacy vs native speedup|New vs native speedup|Oracle vs native speedup|
|---|---:|---:|---:|---:|---:|
|A10|82|1.186x|0.891x|1.056x|1.059x|
|A100|66|1.139x|1.000x|1.139x|1.143x|
|GB200|34|1.021x|1.000x|1.021x|1.063x|
|GB300|33|1.024x|1.000x|1.024x|1.064x|
|H100|45|1.135x|0.895x|1.017x|1.020x|
|L40|40|1.169x|0.968x|1.132x|1.140x|
|RTX PRO 6000|50|1.210x|0.825x|0.998x|1.002x|
|Spark|50|1.268x|0.789x|1.000x|1.001x|
|T4|95|1.113x|0.947x|1.054x|1.055x|
|Thor|35|1.144x|1.000x|1.144x|1.161x|

All datasets were collected with cuDNN frontend version 1.22.1 (`12201`) on
2026-06-12 through 2026-06-30. Most datasets were collected at PyTorch commit
`5140876473ecc91ca14c695d8a40690f4c8919f3` with cuDNN backend version 9.23.0
(`92300`). The GB200 dataset was collected at commit
`f4bdea026bdf3e86cf13216a83d6672b29ebc69e` with cuDNN backend version 9.23.0.
The T4 and A10 datasets were collected at commit
`c7407ac5c3d0ab1a5e1acc66d26de20c8719e828` with cuDNN backend version 9.24.0
(`92400`).

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
the faster measured implementation for each row. They also compare against the
legacy depthwise heuristic from `Convolution.cpp`, excluding rows where the
legacy heuristic and generated heuristic make the same decision.

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
