import argparse
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from gen_data_depthwiseconv import BenchmarkRunnerDepthwiseConv


def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the depthwise cuDNN heuristic check")
    if not torch.backends.cudnn.is_available():
        raise RuntimeError("cuDNN is required for the depthwise cuDNN heuristic check")
    if not hasattr(torch.backends.cudnn, "depthwise_kernel"):
        raise RuntimeError("torch.backends.cudnn.depthwise_kernel is not available")


def depthwise_conv(x, weight, stride):
    padding = weight.shape[-1] // 2
    return F.conv2d(
        x,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=x.shape[1],
    )


def clone_inputs(x, weight):
    x_clone = x.detach().clone().requires_grad_(True)
    weight_clone = weight.detach().clone().requires_grad_(True)
    return x_clone, weight_clone


def compare_dynamic_depthwise_conv(shape, kernel_size, stride):
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(
        shape[1], 1, kernel_size, kernel_size, device="cuda", dtype=torch.float16
    )

    eager_x, eager_weight = clone_inputs(x, weight)
    actual_x, actual_weight = clone_inputs(x, weight)

    torch._dynamo.mark_dynamic(actual_x, 0)
    torch._dynamo.mark_dynamic(actual_x, 2)
    torch._dynamo.mark_dynamic(actual_x, 3)

    compiled = torch.compile(depthwise_conv, dynamic=True)

    expected = depthwise_conv(eager_x, eager_weight, stride)
    expected.float().sum().backward()

    actual = compiled(actual_x, actual_weight, stride)
    actual.float().sum().backward()

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(actual_x.grad, eager_x.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(
        actual_weight.grad, eager_weight.grad, rtol=5e-2, atol=5e-2
    )


def check_symint_depthwise_conv():
    shapes = [
        ((1, 32, 16, 16), 3, 1),
        ((2, 32, 21, 21), 3, 1),
        ((3, 32, 32, 32), 5, 2),
    ]

    with torch.backends.cudnn.flags(
        enabled=True,
        benchmark=True,
        deterministic=False,
        depthwise_kernel="auto",
    ):
        for shape, kernel_size, stride in shapes:
            compare_dynamic_depthwise_conv(shape, kernel_size, stride)
            print(
                "symint conv ok: "
                f"shape={shape}, kernel_size={kernel_size}, stride={stride}"
            )


def check_benchmark_runner(benchmark_iters, warmup_iters):
    runner = BenchmarkRunnerDepthwiseConv()
    runner.nb_iters = benchmark_iters
    runner.nb_warmup_iters = warmup_iters
    runner.args = SimpleNamespace(device="symint-check", grid="train")

    timings = {}
    with torch.backends.cudnn.flags(
        enabled=True,
        benchmark=True,
        deterministic=False,
        depthwise_kernel="auto",
    ):
        for depthwise_kernel in ("native", "cudnn", "auto"):
            torch.backends.cudnn.depthwise_kernel = depthwise_kernel
            fwd_time, bwd_time, all_time = runner.run_benchmark(1, 32, 16, 1, 3)
            for name, value in (
                ("time_fwd", fwd_time),
                ("time_bwd", bwd_time),
                ("time_all", all_time),
            ):
                if not math.isfinite(value):
                    raise AssertionError(
                        f"{name} for {depthwise_kernel} is not finite: {value}"
                    )
            timings[depthwise_kernel] = {
                "time_fwd": fwd_time,
                "time_bwd": bwd_time,
                "time_all": all_time,
            }
            print(f"benchmark ok: depthwise_kernel={depthwise_kernel} {timings[depthwise_kernel]}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "depthwise_symint_check.csv"
        runner.write_metadata(output_file)
        metadata = json.loads(output_file.with_suffix(".metadata.json").read_text())
        required_keys = {
            "benchmark_grid",
            "collection_date",
            "pytorch_commit",
            "device",
            "cudnn_frontend_version",
            "cudnn_backend_version",
        }
        missing_keys = required_keys - metadata.keys()
        if missing_keys:
            raise AssertionError(f"metadata is missing keys: {sorted(missing_keys)}")
        print("metadata ok")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check dynamic SymInt depthwise conv dispatch and the reduced "
            "depthwise AutoHeuristic benchmark path."
        )
    )
    parser.add_argument(
        "--skip-symint",
        action="store_true",
        help="Skip the torch.compile dynamic-shape depthwise convolution check.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the reduced benchmark runner check.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=5,
        help="Benchmark iterations for the reduced benchmark runner check.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Warmup iterations for the reduced benchmark runner check.",
    )
    args = parser.parse_args()

    check_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    if not args.skip_symint:
        check_symint_depthwise_conv()
    if not args.skip_benchmark:
        check_benchmark_runner(args.benchmark_iters, args.warmup_iters)

    print("all checks passed")


if __name__ == "__main__":
    main()
