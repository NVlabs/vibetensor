# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
import sys

KERNELS = [
    {
        "name": "RMSNorm (Medium)",
        "cmd": [
            sys.executable,
            "-m",
            "kernel_factory.rmsnorm.benchmark",
            "--rows",
            "4096",
            "--hidden",
            "4096",
            "--dtype",
            "bfloat16",
        ],
        "regex": r"cutedsl\s+([\d\.]+)",
    },
    {
        "name": "LayerNorm (Medium)",
        "cmd": [
            sys.executable,
            "-m",
            "kernel_factory.layernorm.benchmark",
            "--rows",
            "4096",
            "--hidden",
            "4096",
            "--dtype",
            "bfloat16",
        ],
        "regex": r"cutedsl\s+([\d\.]+)",
    },
    {
        "name": "Rotary (Fused)",
        "cmd": [
            sys.executable,
            "-m",
            "kernel_factory.rotary.benchmark",
            "--batch",
            "4",
            "--heads",
            "32",
            "--seqlen",
            "4096",
            "--headdim",
            "128",
            "--dtype",
            "bfloat16",
        ],
        "regex": r"cutedsl mean\s+:\s+([\d\.]+) ms",
    },
    {
        "name": "Cross Entropy (Large)",
        "cmd": [
            sys.executable,
            "-m",
            "kernel_factory.loss.benchmark",
            "--batch",
            "4",
            "--seq",
            "2048",
            "--vocab",
            "32000",
            "--dtype",
            "bfloat16",
        ],
        "regex": r"cutedsl mean\s+:\s+([\d\.]+) ms",
    },
]


def run_benchmarks():
    print(f"{'Kernel':<25} | {'CuTeDSL Time (ms)':<18} | {'Status':<10}")
    print("-" * 60)

    env = os.environ.copy()
    env["PYTHONPATH"] = "tmp"

    for kernel in KERNELS:
        try:
            result = subprocess.run(
                kernel["cmd"], capture_output=True, text=True, env=env
            )

            if result.returncode != 0:
                print(f"{kernel['name']:<25} | {'N/A':<18} | FAILED")
                print(f"Error output:\n{result.stderr}")
                continue

            output = result.stdout
            match = re.search(kernel["regex"], output)
            if match:
                time_ms = match.group(1)
                print(f"{kernel['name']:<25} | {time_ms:<18} | PASS")
            else:
                print(f"{kernel['name']:<25} | {'N/A':<18} | PARSE ERR")
                # print(output)

        except Exception as e:
            print(f"{kernel['name']:<25} | {'N/A':<18} | ERROR: {e}")


if __name__ == "__main__":
    run_benchmarks()
