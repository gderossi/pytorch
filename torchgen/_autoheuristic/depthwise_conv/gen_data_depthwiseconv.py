import argparse
import itertools
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
from torch._inductor.runtime.benchmarking import benchmarker


REPO_ROOT = Path(__file__).resolve().parents[3]


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return getattr(torch.version, "git_version", None)


def get_cudnn_frontend_version() -> int | None:
    version_header = (
        REPO_ROOT / "third_party/cudnn_frontend/include/cudnn_frontend_version.h"
    )
    if not version_header.exists():
        return None

    versions: dict[str, int] = {}
    pattern = re.compile(r"#define CUDNN_FRONTEND_(MAJOR|MINOR|PATCH)_VERSION (\d+)")
    for line in version_header.read_text().splitlines():
        match = pattern.match(line)
        if match:
            versions[match.group(1)] = int(match.group(2))

    if {"MAJOR", "MINOR", "PATCH"} <= versions.keys():
        return versions["MAJOR"] * 10000 + versions["MINOR"] * 100 + versions["PATCH"]
    return None


class BenchmarkRunnerDepthwiseConv:
    def __init__(self):
        self.train_batch_sizes = [1, 8, 32]
        self.train_in_channels = [32, 96, 384, 1024]
        self.train_shapes = [
            (7, 7),
            (14, 14),
            (28, 28),
            (56, 56),
            (112, 112),
            (32, 32),
            (64, 64),
            (128, 128),
            (10, 21),
            (21, 10),
            (30, 44),
            (44, 30),
            (44, 88),
            (88, 44),
        ]
        self.train_extra_batch_sizes = [4, 16, 64]
        self.train_extra_in_channels = [64, 128, 256, 512]
        self.train_extra_shapes = [
            (7, 7),
            (14, 14),
            (28, 28),
            (56, 56),
            (112, 112),
            (32, 32),
            (64, 64),
            (128, 128),
        ]
        self.validation_batch_sizes = [3, 12, 48]
        self.validation_in_channels = [48, 192, 768]
        self.validation_shapes = [
            (10, 10),
            (21, 21),
            (30, 30),
            (44, 44),
            (88, 88),
            (10, 21),
            (21, 10),
            (30, 44),
            (44, 30),
            (44, 88),
            (88, 44),
        ]
        self.test_batch_sizes = [5, 20]
        self.test_in_channels = [40, 160, 640]
        self.test_shapes = [
            (13, 13),
            (26, 26),
            (52, 52),
            (104, 104),
            (13, 26),
            (26, 13),
            (32, 96),
            (96, 32),
        ]
        self.batch_sizes = self.train_batch_sizes
        self.in_channels = self.train_in_channels
        self.shapes = self.train_shapes
        self.strides = [1, 2]
        self.kernel_sizes = [1, 3, 5]
        self.workload_cases = []

        self.nb_warmup_iters = 50
        self.nb_iters = 100

        self.columns = [
            "sm",
            "bs",
            "ch",
            "h",
            "w",
            "filter",
            "stride",
            "time_fwd",
            "time_bwd",
            "time_all",
            "time_fwd_cudnn",
            "time_bwd_cudnn",
            "time_all_cudnn",
            "cudnn_speedup_fwd",
            "cudnn_speedup_bwd",
            "cudnn_speedup_all",
        ]

        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

        major, minor = torch.cuda.get_device_capability()
        self.sm = major * 10 + minor

    def add_base_arguments(self):
        self.parser.add_argument(
            "--device",
            type=str,
            default="",
            help="Label for device being benchmarked",
        )
        self.parser.add_argument(
            "--output",
            type=str,
            default="",
            help="Output CSV path. Defaults to data_depthwiseconv_[GRID]_[DEVICE].csv",
        )
        self.parser.add_argument(
            "--grid",
            type=str,
            default="train",
            choices=["train", "train-extra", "validation", "test"],
            help="Benchmark grid to run (default: train)",
        )
        self.parser.add_argument(
            "--benchmark-iters",
            type=int,
            default=self.nb_iters,
            help=f"Benchmark iterations per shape (default: {self.nb_iters})",
        )
        self.parser.add_argument(
            "--warmup-iters",
            type=int,
            default=self.nb_warmup_iters,
            help=f"Warmup iterations per shape (default: {self.nb_warmup_iters})",
        )

    def parse_args(self):
        return self.parser.parse_args()

    def build_workload_cases(self, batch_sizes, in_channels, shapes):
        return [
            (batch_size, c, h, w, s, k)
            for batch_size, c, (h, w), s, k in itertools.product(
                batch_sizes,
                in_channels,
                shapes,
                self.strides,
                self.kernel_sizes,
            )
        ]

    def select_grid(self):
        if self.args.grid == "train":
            self.batch_sizes = self.train_batch_sizes
            self.in_channels = self.train_in_channels
            self.shapes = self.train_shapes
            self.workload_cases = self.build_workload_cases(
                self.train_batch_sizes, self.train_in_channels, self.train_shapes
            )
            self.workload_cases += self.build_workload_cases(
                self.train_extra_batch_sizes,
                self.train_extra_in_channels,
                self.train_extra_shapes,
            )
        elif self.args.grid == "train-extra":
            self.batch_sizes = self.train_extra_batch_sizes
            self.in_channels = self.train_extra_in_channels
            self.shapes = self.train_extra_shapes
            self.workload_cases = self.build_workload_cases(
                self.batch_sizes, self.in_channels, self.shapes
            )
        elif self.args.grid == "validation":
            self.batch_sizes = self.validation_batch_sizes
            self.in_channels = self.validation_in_channels
            self.shapes = self.validation_shapes
            self.workload_cases = self.build_workload_cases(
                self.batch_sizes, self.in_channels, self.shapes
            )
        else:
            self.batch_sizes = self.test_batch_sizes
            self.in_channels = self.test_in_channels
            self.shapes = self.test_shapes
            self.workload_cases = self.build_workload_cases(
                self.batch_sizes, self.in_channels, self.shapes
            )

    def run_benchmark(self, batch_size, c, h, w, s, k):
        # Note: cuDNN depthwise conv only supports FP16
        x = torch.randn(
            batch_size, c, h, w, device="cuda", dtype=torch.half, requires_grad=True
        )

        pad = k // 2
        conv = (
            nn.Conv2d(
                in_channels=c,
                out_channels=c,
                kernel_size=k,
                stride=s,
                padding=pad,
                groups=c,
                bias=False,
            )
            .half()
            .to("cuda")
        )

        print(
            "Testing [N, C, H, W]=[{}, {}, {}, {}], kH/kW={}, stride={}, pad={}, depthwise_kernel={}".format(
                *x.size(), k, s, pad, torch.backends.cudnn.depthwise_kernel
            )
        )

        # Perform some dummy iterations to warmup cudnn.benchmark
        for _ in range(self.nb_warmup_iters):
            output = conv(x)

        # Perform warmup for backwards
        g0 = torch.rand_like(output)
        for _ in range(self.nb_warmup_iters):
            output = conv(x)
            x.grad = None
            conv.weight.grad = None
            output.backward(g0)

        def run_forward():
            return conv(x)

        def run_forward_backward():
            output = conv(x)
            output.backward(g0)

        fwd_time = benchmarker.benchmark(
            run_forward,
            device="cuda",
            benchmark_iters=self.nb_iters,
            is_vetted_benchmarking=True,
        )
        all_time = benchmarker.benchmark(
            run_forward_backward,
            device="cuda",
            benchmark_iters=self.nb_iters,
            grad_to_none=[x, conv.weight],
            is_vetted_benchmarking=True,
        )
        bwd_time = all_time - fwd_time
        return fwd_time, bwd_time, all_time

    def write_metadata(self, output_file):
        device_index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
        metadata = {
            "benchmark_grid": {
                "name": self.args.grid,
                "batch_sizes": self.batch_sizes,
                "in_channels": self.in_channels,
                "shapes": self.shapes,
                "strides": self.strides,
                "kernel_sizes": self.kernel_sizes,
                "workload_count": len(self.workload_cases),
                "warmup_iters": self.nb_warmup_iters,
                "benchmark_iters": self.nb_iters,
            },
            "collection_date": datetime.now(timezone.utc).date().isoformat(),
            "pytorch_commit": get_git_commit(),
            "device": {
                "label": self.args.device,
                "name": torch.cuda.get_device_name(device_index),
                "index": device_index,
                "type": "cuda",
                "sm": major * 10 + minor,
            },
            "cudnn_frontend_version": get_cudnn_frontend_version(),
            "cudnn_backend_version": torch.backends.cudnn.version(),
        }
        metadata_file = Path(output_file).with_suffix(".metadata.json")
        metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")

    def run(self):
        self.args = self.parse_args()
        self.nb_iters = self.args.benchmark_iters
        self.nb_warmup_iters = self.args.warmup_iters
        self.select_grid()
        if self.args.device == "":
            self.args.device = torch.cuda.get_device_name().replace(" ", "-")

        results = pd.DataFrame()

        for batch_size, c, h, w, s, k in self.workload_cases:
            torch.backends.cudnn.depthwise_kernel = "native"
            fwd_time, bwd_time, all_time = self.run_benchmark(
                batch_size, c, h, w, s, k
            )

            torch.backends.cudnn.depthwise_kernel = "cudnn"
            fwd_time_cudnn, bwd_time_cudnn, all_time_cudnn = self.run_benchmark(
                batch_size, c, h, w, s, k
            )

            cudnn_speedup_fwd = fwd_time / fwd_time_cudnn
            cudnn_speedup_bwd = bwd_time / bwd_time_cudnn
            cudnn_speedup_all = all_time / all_time_cudnn

            tmp_df = pd.DataFrame(
                [
                    [
                        self.sm,
                        batch_size,
                        c,
                        h,
                        w,
                        k,
                        s,
                        fwd_time,
                        bwd_time,
                        all_time,
                        fwd_time_cudnn,
                        bwd_time_cudnn,
                        all_time_cudnn,
                        cudnn_speedup_fwd,
                        cudnn_speedup_bwd,
                        cudnn_speedup_all,
                    ]
                ],
                columns=self.columns,
            )
            results = pd.concat([results, tmp_df])

        output_file = self.args.output or (
            f"data_depthwiseconv_{self.args.grid}_{self.args.device}.csv"
        )
        results.to_csv(output_file, index=False)
        self.write_metadata(output_file)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    runner = BenchmarkRunnerDepthwiseConv()
    runner.run()
