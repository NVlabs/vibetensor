# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest


@pytest.mark.parametrize("file_rel", ["src/vbt/dispatch/plugin_loader.cc"]) 
def test_loader_uses_allowed_rtld_flags_and_no_deepbind(file_rel: str):
    repo = pathlib.Path(__file__).resolve().parents[2]
    src = repo / file_rel
    txt = src.read_text(encoding="utf-8")
    # Ensure flags literals exist at the dlopen call sites
    assert "RTLD_NOW" in txt and "RTLD_LOCAL" in txt
    # Disallow deepbind/global anywhere in loader
    assert "RTLD_DEEPBIND" not in txt and "RTLD_GLOBAL" not in txt


def _scan_repo_tokens(root: pathlib.Path, tokens: list[str]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}

    # CI may install Python deps under the repo root (e.g. repo/.local).
    # We only want to scan *this* repository's source code.
    skip_dir_parts = {
        "3rdparty",
        "tests",
        ".ref",
        ".git",
        ".venv",
        ".local",
        "build",
        "build-py",
        "build-full",
        "dist",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".matrixfoundry",
    }

    for pat in ("*.c", "*.cc", "*.cpp", "*.h", "*.hpp"):
        for p in root.rglob(pat):
            try:
                rel_parts = p.relative_to(root).parts
            except ValueError:
                rel_parts = p.parts
            if any(part in skip_dir_parts for part in rel_parts):
                continue

            sp = str(p)
            try:
                t = p.read_text(encoding="utf-8")
            except Exception:
                continue
            for tok in tokens:
                if tok in t:
                    hits.setdefault(tok, []).append(sp)
    return hits


def test_repo_has_no_deepbind_global_tokens():
    repo = pathlib.Path(__file__).resolve().parents[2]
    hits = _scan_repo_tokens(repo, ["RTLD_DEEPBIND", "RTLD_GLOBAL"])
    assert hits.get("RTLD_DEEPBIND") is None, f"Found RTLD_DEEPBIND in: {hits.get('RTLD_DEEPBIND')}"
    # We allow RTLD_GLOBAL nowhere in src; should be absent
    assert hits.get("RTLD_GLOBAL") is None, f"Found RTLD_GLOBAL in: {hits.get('RTLD_GLOBAL')}"
