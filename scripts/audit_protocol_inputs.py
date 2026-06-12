from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()

CANDIDATE_DIRS = [
    "data",
    "cache",
    "outputs",
    "runs",
    "reports",
]

PATTERNS = [
    "*.npy",
    "*.npz",
    "*.parquet",
    "*.csv",
    "*.json",
    "*.jsonl",
    "*.pt",
    "*.pth",
    "*.pkl",
]

KEYWORDS = [
    "traj",
    "trajectory",
    "seq",
    "sequence",
    "frame",
    "index",
    "sample",
    "current",
    "future",
    "pose",
    "action",
    "error",
    "gain",
    "cheap",
    "exp",
    "expensive",
    "dino",
    "latent",
]


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def describe_npy(path: Path) -> str:
    try:
        import numpy as np
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return f"npy shape={arr.shape}, dtype={arr.dtype}"
    except Exception as exc:
        return f"npy unreadable: {type(exc).__name__}: {exc}"


def describe_npz(path: Path) -> str:
    try:
        import numpy as np
        data = np.load(path, allow_pickle=False)
        parts = []
        for k in list(data.files)[:20]:
            arr = data[k]
            parts.append(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
        more = "" if len(data.files) <= 20 else f" ... +{len(data.files)-20} keys"
        return "npz " + "; ".join(parts) + more
    except Exception as exc:
        return f"npz unreadable: {type(exc).__name__}: {exc}"


def describe_parquet(path: Path) -> str:
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = pf.schema.names
        useful = [c for c in cols if any(k in c.lower() for k in KEYWORDS)]
        return (
            f"parquet rows={pf.metadata.num_rows}, cols={len(cols)}, "
            f"useful_cols={useful[:30]}"
        )
    except Exception as exc:
        return f"parquet unreadable: {type(exc).__name__}: {exc}"


def describe_csv(path: Path) -> str:
    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=5)
        cols = list(df.columns)
        useful = [c for c in cols if any(k in c.lower() for k in KEYWORDS)]
        return f"csv cols={len(cols)}, useful_cols={useful[:30]}"
    except Exception as exc:
        return f"csv unreadable: {type(exc).__name__}: {exc}"


def describe_json(path: Path) -> str:
    try:
        text = path.read_text(errors="replace")
        if path.suffix == ".jsonl":
            lines = text.splitlines()
            first = json.loads(lines[0]) if lines else {}
            if isinstance(first, dict):
                useful = [k for k in first.keys() if any(s in k.lower() for s in KEYWORDS)]
                return f"jsonl lines~={len(lines)}, first_keys={list(first.keys())[:30]}, useful_keys={useful[:30]}"
            return f"jsonl lines~={len(lines)}, first_type={type(first).__name__}"
        obj = json.loads(text)
        if isinstance(obj, dict):
            useful = [k for k in obj.keys() if any(s in k.lower() for s in KEYWORDS)]
            return f"json dict keys={list(obj.keys())[:30]}, useful_keys={useful[:30]}"
        if isinstance(obj, list):
            first = obj[0] if obj else None
            if isinstance(first, dict):
                useful = [k for k in first.keys() if any(s in k.lower() for s in KEYWORDS)]
                return f"json list len={len(obj)}, first_keys={list(first.keys())[:30]}, useful_keys={useful[:30]}"
            return f"json list len={len(obj)}, first_type={type(first).__name__}"
        return f"json type={type(obj).__name__}"
    except Exception as exc:
        return f"json unreadable: {type(exc).__name__}: {exc}"


def describe(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return describe_npy(path)
    if suffix == ".npz":
        return describe_npz(path)
    if suffix == ".parquet":
        return describe_parquet(path)
    if suffix == ".csv":
        return describe_csv(path)
    if suffix in {".json", ".jsonl"}:
        return describe_json(path)
    if suffix in {".pt", ".pth", ".pkl"}:
        return "binary checkpoint/cache; not loaded for safety"
    return ""


def main() -> None:
    print("# SPSM protocol input audit\n")
    print(f"root: `{ROOT}`\n")

    files = []
    for d in CANDIDATE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for pattern in PATTERNS:
            files.extend(base.rglob(pattern))

    files = sorted(set(files), key=lambda p: (str(p.parent), p.name))

    print(f"Found {len(files)} candidate files.\n")
    print("| path | size MB | description |")
    print("| --- | ---: | --- |")

    for path in files:
        rel = path.relative_to(ROOT)
        mb = size_mb(path)
        desc = describe(path)
        desc = desc.replace("\n", " ")
        print(f"| `{rel}` | {mb:.2f} | {desc} |")

    print("\n# Likely protocol-relevant files\n")
    for path in files:
        name = str(path).lower()
        if any(k in name for k in KEYWORDS):
            print(f"- `{path.relative_to(ROOT)}` ({size_mb(path):.2f} MB)")


if __name__ == "__main__":
    main()
