import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


REPO_ROOT = Path(__file__).resolve().parents[3]


class TrainDecisionTreeDepthwiseConv:
    def __init__(self):
        self.sm_feature = "sm"
        self.raw_features = ["bs", "ch", "h", "w", "filter", "stride"]
        self.derived_features = ["out_h", "out_w", "out_elements", "kernel_work"]
        self.features = [*self.raw_features, *self.derived_features]
        self.target = "cudnn_speedup_all"
        self.classes = ["false", "true"]
        self.sm_groups = [75, 80, 86, 89, 90, 100, 120]
        self.output_file = (
            REPO_ROOT / "aten/src/ATen/autoheuristic/DepthwiseConvHeuristic.h"
        )
        self.opt_name = "depthwise_conv"
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

    def add_base_arguments(self):
        # Data parameters
        self.parser.add_argument(
            "input_files", type=str, nargs="+", help="Paths to processed CSV files"
        )
        self.parser.add_argument(
            "--tolerance",
            type=float,
            default=0.0,
            help="Tolerance threshold (default: 0.0)",
        )

        # Model parameters
        self.parser.add_argument(
            "--max-depth",
            type=int,
            default=None,
            help="Maximum tree depth (default: None = unlimited)",
        )
        self.parser.add_argument(
            "--max-leaf-nodes",
            type=int,
            default=None,
            help="Maximum number of leaf nodes (default: None = unlimited)",
        )
        self.parser.add_argument(
            "--min-samples-split",
            type=int,
            default=2,
            help="Minimum samples to split a node (default: 2)",
        )
        self.parser.add_argument(
            "--min-samples-leaf",
            type=int,
            default=1,
            help="Minimum samples in leaf node (default: 1)",
        )
        self.parser.add_argument(
            "--criterion",
            type=str,
            default="gini",
            choices=["gini", "entropy", "log_loss"],
            help="Split criterion (default: gini)",
        )

        # Other
        self.parser.add_argument(
            "--seed", type=int, default=42, help="Random seed (default: 42)"
        )
        self.parser.add_argument(
            "--output-file",
            type=Path,
            default=self.output_file,
            help=f"Output header path (default: {self.output_file})",
        )
        self.parser.add_argument(
            "--validation-files",
            type=str,
            nargs="+",
            default=[],
            help="Optional holdout CSV files used only for validation",
        )
        self.parser.add_argument(
            "--test-files",
            type=str,
            nargs="+",
            default=[],
            help="Optional final holdout CSV files used only for reporting",
        )

    def parse_args(self):
        return self.parser.parse_args()

    def cudnn_sm_group(self, sm):
        if sm < 80:
            return 75
        if sm < 86:
            return 80
        if sm < 89:
            return 86
        if sm < 90:
            return 89
        if sm < 100:
            return 90
        if sm < 120:
            return 100
        return 120

    def nearest_generated_group(self, sm_group, generated_sm_groups):
        return min(generated_sm_groups, key=lambda group: abs(group - sm_group))

    def add_derived_features(self, df):
        df = df.copy()
        if "h" not in df.columns:
            df["h"] = df["w"]
        else:
            df["h"] = df["h"].fillna(df["w"])
        df["out_h"] = (
            (df["h"] + 2 * (df["filter"] // 2) - df["filter"]) // df["stride"]
        ) + 1
        df["out_w"] = (
            (df["w"] + 2 * (df["filter"] // 2) - df["filter"]) // df["stride"]
        ) + 1
        df["out_elements"] = df["bs"] * df["ch"] * df["out_h"] * df["out_w"]
        df["kernel_work"] = df["out_elements"] * df["filter"] * df["filter"]
        return df

    def load_and_prepare_data(self, input_files, tolerance):
        """
        Load data and prepare for binary classification.
        Filters out label 2 (equal performance within tolerance).
        """
        required_columns = [
            self.sm_feature,
            "bs",
            "ch",
            "w",
            "filter",
            "stride",
            self.target,
        ]
        dfs = []

        for input_file in input_files:
            if not input_file.endswith(".csv"):
                raise ValueError(
                    f"Invalid file format: {input_file}. Expected CSV file with .csv extension."
                )

            try:
                df_full = pd.read_csv(input_file)
            except Exception as e:
                raise ValueError(f"Failed to read CSV file {input_file}: {e}") from e

            missing_columns = set(required_columns) - set(df_full.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in {input_file}: {sorted(missing_columns)}. "
                    f"Required columns: {required_columns}"
                )
            optional_columns = ["h"] if "h" in df_full.columns else []
            df = df_full[[*required_columns, *optional_columns]]

            if df.isnull().any().any():
                empty_cols = df.columns[df.isnull().any()].tolist()
                raise ValueError(
                    f"File {input_file} contains empty cells in columns: {empty_cols}."
                )

            dfs.append(df)

        df = self.add_derived_features(pd.concat(dfs, ignore_index=True))

        sm_values = df[self.sm_feature].values
        features = df[self.features].values
        speedup_values = df[self.target].values
        sample_weights = abs(speedup_values - 1.0) - tolerance

        # Create original labels (0, 1, 2)
        lower_tolerance = 1.0 - tolerance
        upper_tolerance = 1.0 + tolerance
        labels = np.zeros(len(speedup_values), dtype=np.int64)
        labels[speedup_values < lower_tolerance] = 0
        labels[speedup_values > upper_tolerance] = 1
        labels[
            (speedup_values >= lower_tolerance) & (speedup_values <= upper_tolerance)
        ] = 2

        # Filter out label 2 (results within tolerance)
        mask = labels != 2
        features = features[mask]
        labels = labels[mask]
        sample_weights = sample_weights[mask]
        sm_values = sm_values[mask]
        sm_groups = np.array(
            [self.cudnn_sm_group(sm) for sm in sm_values], dtype=np.int64
        )

        return features, labels, sample_weights, sm_groups

    def create_decision_tree(self, args, features, labels, sample_weights, name):
        # Create and train decision tree
        print(f"Model parameters for {name}:")
        print(f"  Criterion: {args.criterion}")
        print(
            f"  Max depth: {args.max_depth if args.max_depth else 'None (unlimited)'}"
        )
        print(
            f"  Max leaf nodes: {args.max_leaf_nodes if args.max_leaf_nodes else 'None (unlimited)'}"
        )
        print(f"  Min samples split: {args.min_samples_split}")
        print(f"  Min samples leaf: {args.min_samples_leaf}")
        print()

        model = DecisionTreeClassifier(
            criterion=args.criterion,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_leaf_nodes=args.max_leaf_nodes,
            random_state=args.seed,
        )

        model.fit(features, labels, sample_weight=sample_weights)

        print(f"✓ Training complete for {name}!")
        print("\nTree statistics:")
        print(f"  Total nodes: {model.tree_.node_count}")
        print(f"  Leaves: {model.tree_.n_leaves}")
        print(f"  Max depth reached: {model.tree_.max_depth}")

        # Evaluate on training data
        predictions = model.predict(features)
        accuracy = accuracy_score(labels, predictions)
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        return model

    def evaluate_validation(
        self,
        name,
        validation_files,
        tolerance,
        models,
        generated_sm_groups,
    ):
        features, labels, _sample_weights, sm_groups = self.load_and_prepare_data(
            input_files=validation_files, tolerance=tolerance
        )
        predictions = np.empty_like(labels)

        print(f"{name} results:")
        for sm_group in self.sm_groups:
            mask = sm_groups == sm_group
            if not mask.any():
                continue

            target_group = self.nearest_generated_group(sm_group, generated_sm_groups)
            model = models[target_group]
            group_predictions = model.predict(features[mask])
            predictions[mask] = group_predictions
            accuracy = accuracy_score(labels[mask], group_predictions)
            matrix = confusion_matrix(
                labels[mask], group_predictions, labels=[0, 1]
            )
            print(
                f"  cuDNN SM group {sm_group} "
                f"(tree sm{target_group}): "
                f"accuracy={accuracy:.4f} ({accuracy * 100:.2f}%), "
                f"confusion_matrix=[[{matrix[0, 0]}, {matrix[0, 1]}], "
                f"[{matrix[1, 0]}, {matrix[1, 1]}]]"
            )

        accuracy = accuracy_score(labels, predictions)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        print(
            f"  overall: accuracy={accuracy:.4f} ({accuracy * 100:.2f}%), "
            f"confusion_matrix=[[{matrix[0, 0]}, {matrix[0, 1]}], "
            f"[{matrix[1, 0]}, {matrix[1, 1]}]]"
        )
        self.report_policy_metrics(validation_files, tolerance, labels, predictions)

    def report_policy_metrics(self, input_files, tolerance, labels, predictions):
        dfs = []
        for input_file in input_files:
            dfs.append(pd.read_csv(input_file))
        df = self.add_derived_features(pd.concat(dfs, ignore_index=True))

        speedup_values = df[self.target].values
        lower_tolerance = 1.0 - tolerance
        upper_tolerance = 1.0 + tolerance
        raw_labels = np.zeros(len(speedup_values), dtype=np.int64)
        raw_labels[speedup_values < lower_tolerance] = 0
        raw_labels[speedup_values > upper_tolerance] = 1
        raw_labels[
            (speedup_values >= lower_tolerance) & (speedup_values <= upper_tolerance)
        ] = 2
        df = df[raw_labels != 2]
        if len(df) != len(predictions) or len(labels) != len(predictions):
            raise RuntimeError("Policy metric rows do not match predictions")

        native = df["time_all"].values
        cudnn = df["time_all_cudnn"].values
        chosen = np.where(predictions == 1, cudnn, native)
        oracle = np.minimum(native, cudnn)
        always_cudnn = cudnn
        slowdown = chosen / oracle

        print(
            "  policy_metrics: "
            f"heuristic_total_speedup={native.sum() / chosen.sum():.4f}, "
            f"always_cudnn_total_speedup={native.sum() / always_cudnn.sum():.4f}, "
            f"oracle_total_speedup={native.sum() / oracle.sum():.4f}, "
            f"total_slowdown_vs_oracle={chosen.sum() / oracle.sum():.4f}, "
            f"max_slowdown_vs_oracle={slowdown.max():.4f}"
        )

    def is_leaf_node(self, tree, node_id):
        """Check if a node is a leaf node."""
        return tree.children_left[node_id] == tree.children_right[node_id] == -1

    def get_leaf_class(self, tree, node_id):
        """Get the class label for a leaf node."""
        class_values = tree.value[node_id][0]
        return self.classes[np.argmax(class_values)]

    def codegen_header(self):
        header = (
            f"""// This file was generated by AutoHeuristic. Do not modify it manually!
// To regenerate this file, take a look at the README.md file inside torchgen/_autoheuristic/{self.opt_name}/"""
            + """
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>

namespace at::native {
"""
        )
        return header

    def codegen_footer(self):
        return "\n} // namespace at::native\n"

    def codegen_tree_header(self, sm_group):
        return f"""
template <typename T>
static bool check_cudnn_depthwise_workload_sm{sm_group}(
    T stride, T filter, T w, T ch, T bs, T h, T out_h, T out_w,
    T out_elements, T kernel_work) {{
  // auto-generated heuristic decision tree for cuDNN SM group {sm_group}"""

    def codegen_dispatch(self, lines: list[str], generated_sm_groups: list[int]):
        lines.append(
            """
template <typename T>
static bool check_cudnn_depthwise_workload_with_filter(
    const at::Tensor& input, T stride, const at::Tensor& weight) {
  const int sm = at::detail::getCUDAHooks().getDeviceCapability(
      input.get_device());
  TORCH_INTERNAL_ASSERT(sm != 0, "CUDA not available");

  // 1D conv
  if (at::symint::size<T>(input, 2) == 1 && stride == 1) {
    return true;
  }
  // 2D conv
  // only 1/2 stride
  if (stride != 1 && stride != 2) return false;
  // only square filters
  if (at::symint::size<T>(weight, 2) !=
      at::symint::size<T>(weight, 3)) {
    return false;
  }
  auto filter = at::symint::size<T>(weight, 3);
  // only 1/3/5 filter
  if (filter != 1 && filter != 3 && filter != 5) return false;
  auto h = at::symint::size<T>(input, 2);
  auto w = at::symint::size<T>(input, 3);
  if (h < 7) return false;
  if (w < 7) return false;
  auto ch = at::symint::size<T>(input, 1);
  auto bs = at::symint::size<T>(input, 0);
  auto out_h = (h + 2 * (filter / 2) - filter) / stride + 1;
  auto out_w = (w + 2 * (filter / 2) - filter) / stride + 1;
  auto out_elements = bs * ch * out_h * out_w;
  auto kernel_work = out_elements * filter * filter;
"""
        )
        dispatch_ranges = [
            ("sm < 80", 75),
            ("sm < 86", 80),
            ("sm < 89", 86),
            ("sm < 90", 89),
            ("sm < 100", 90),
            ("sm < 120", 100),
        ]
        for condition, sm_group in dispatch_ranges:
            target_group = self.nearest_generated_group(
                sm_group, generated_sm_groups
            )
            lines.extend(
                [
                    f"  if ({condition}) return "
                    f"check_cudnn_depthwise_workload_sm{target_group}<T>(",
                    "      stride, filter, w, ch, bs, h, out_h, out_w,",
                    "      out_elements, kernel_work);",
                ]
            )
        target_group = self.nearest_generated_group(120, generated_sm_groups)
        lines.extend(
            [
                f"  return check_cudnn_depthwise_workload_sm{target_group}<T>(",
                "      stride, filter, w, ch, bs, h, out_h, out_w,",
                "      out_elements, kernel_work);",
            ]
        )
        lines.append("}\n")

    def codegen(self, tree, lines: list[str]):
        feature_names = self.features

        def codegen_node(node_id, depth):
            """Recursively traverse tree nodes and generate C++ code."""
            indent = "  " * (depth + 1)

            # Handle leaf nodes
            if self.is_leaf_node(tree, node_id):
                class_label = self.get_leaf_class(tree, node_id)
                lines.append(f"{indent}return {class_label};")
                return

            # Internal node - generate condition
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]

            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            # Check if children are leaf nodes for inline formatting
            left_is_leaf = self.is_leaf_node(tree, left_child)
            right_is_leaf = self.is_leaf_node(tree, right_child)

            cpp_condition = f"{feature_name} <= {int(threshold)}"

            if left_is_leaf and right_is_leaf:
                # Both children are leaves - use inline format
                left_class = self.get_leaf_class(tree, left_child)
                right_class = self.get_leaf_class(tree, right_child)
                # Leaves with the same class can be combined
                if left_class == right_class:
                    lines.append(f"{indent}return {left_class};")
                else:
                    lines.append(f"{indent}if ({cpp_condition}) return {left_class};")
                    lines.append(f"{indent}else return {right_class};")
            elif left_is_leaf:
                # Only left child is leaf - inline if, else block
                left_class = self.get_leaf_class(tree, left_child)
                lines.append(f"{indent}if ({cpp_condition}) return {left_class};")
                lines.append(f"{indent}else {{")
                codegen_node(right_child, depth + 1)
                lines.append(f"{indent}}}")
            elif right_is_leaf:
                # Only right child is leaf - if block, inline else
                right_class = self.get_leaf_class(tree, right_child)
                lines.append(f"{indent}if ({cpp_condition}) {{")
                codegen_node(left_child, depth + 1)
                lines.append(f"{indent}}}")
                lines.append(f"{indent}else return {right_class};")
            else:
                # Both children are internal nodes - full if-else blocks
                lines.append(f"{indent}if ({cpp_condition}) {{")
                codegen_node(left_child, depth + 1)
                lines.append(f"{indent}}}")
                lines.append(f"{indent}else {{")
                codegen_node(right_child, depth + 1)
                lines.append(f"{indent}}}")

        codegen_node(0, 0)

    def write_heuristic_to_file(self, lines: list[str]):
        with open(self.args.output_file, "w") as f:
            f.write("\n".join(lines))

    def generate_heuristic(self):
        self.args = self.parse_args()

        # Set random seed
        np.random.seed(self.args.seed)

        # Load data
        features, labels, sample_weights, sm_groups = self.load_and_prepare_data(
            input_files=self.args.input_files, tolerance=self.args.tolerance
        )

        lines = [self.codegen_header()]
        generated_sm_groups = []
        models = {}
        for sm_group in self.sm_groups:
            mask = sm_groups == sm_group
            if not mask.any():
                print(f"Skipping cuDNN SM group {sm_group}: no samples")
                continue
            generated_sm_groups.append(sm_group)
            model = self.create_decision_tree(
                self.args,
                features[mask],
                labels[mask],
                sample_weights[mask],
                f"cuDNN SM group {sm_group}",
            )
            models[sm_group] = model

            tree_header = self.codegen_tree_header(sm_group)
            tree_header += ", tolerance = " + str(self.args.tolerance)
            lines.append(tree_header)
            self.codegen(model.tree_, lines)
            lines.append("}\n")

        if not generated_sm_groups:
            raise RuntimeError("No cuDNN SM groups had training samples")

        self.codegen_dispatch(lines, generated_sm_groups)
        lines.append(self.codegen_footer())
        self.write_heuristic_to_file(lines)

        if self.args.validation_files:
            self.evaluate_validation(
                "Validation",
                self.args.validation_files,
                self.args.tolerance,
                models,
                generated_sm_groups,
            )
        if self.args.test_files:
            self.evaluate_validation(
                "Test",
                self.args.test_files,
                self.args.tolerance,
                models,
                generated_sm_groups,
            )


if __name__ == "__main__":
    train = TrainDecisionTreeDepthwiseConv()
    train.generate_heuristic()
