from __future__ import annotations

from pathlib import Path
import json
import numpy as np

ROOT = Path.cwd()

NEEDLES = [
    "tartanair_dinov2_features",
    "features_train.npz",
    "features_val.npz",
    "features.npz",
    "z_current",
    "z_future",
    "np.savez",
    "savez",
    "train_test_split",
    "random_split",
    "shuffle",
    "TartanAir",
    "tartanair",
    "JapaneseAlley",
    "trajectory",
    "traj",
    "frame",
    "pose",
    "image_left",
]

TEXT_EXTS = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json", ".tex"}

SEARCH_DIRS = [
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "reports",
    ROOT,
]

NPZ_PATHS = [
    ROOT / "outputs/tartanair_dinov2_features/features.npz",
    ROOT / "outputs/tartanair_dinov2_features/features_train.npz",
    ROOT / "outputs/tartanair_dinov2_features/features_val.npz",
    ROOT / "outputs/tartanair_dinov2_features/features_train_errorrel.npz",
    ROOT / "outputs/tartanair_dinov2_features/features_val_errorrel.npz",
    ROOT / "outputs/tartanair_resnet18_features_full/features.npz",
    ROOT / "outputs/tartanair_resnet18_features_full/features_train.npz",
    ROOT / "outputs/tartanair_resnet18_features_full/features_val.npz",
]


def is_inside(path: Path, name: str) -> bool:
    return any(part == name for part in path.parts)


def iter_text_files():
    seen = set()
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        if base.is_file():
            files = [base]
        else:
            files = list(base.rglob("*"))

        for p in files:
            if p in seen:
                continue
            seen.add(p)

            if not p.is_file():
                continue
            if is_inside(p, ".git") or is_inside(p, "__pycache__"):
                continue
            if is_inside(p, "outputs") and p.suffix != ".json":
                continue
            if p.suffix.lower() not in TEXT_EXTS:
                continue
            yield p


def print_matching_lines(path: Path, lines: list[str], max_hits: int = 30) -> None:
    hits = []
    lower_needles = [n.lower() for n in NEEDLES]

    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(n in low for n in lower_needles):
            hits.append((i, line.rstrip("\n")))

    if not hits:
        return

    print(f"\n## {path.relative_to(ROOT)}")
    for i, line in hits[:max_hits]:
        clean = line.replace("|", "\\|")
        print(f"- L{i}: `{clean[:220]}`")
    if len(hits) > max_hits:
        print(f"- ... {len(hits) - max_hits} more hits")


def audit_code() -> None:
    print("# Feature provenance audit\n")
    print("## Source-code / text matches")

    for path in sorted(iter_text_files()):
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        lines = text.splitlines()
        print_matching_lines(path, lines)


def describe_npz(path: Path) -> None:
    print(f"\n## {path.relative_to(ROOT)}")

    if not path.exists():
        print("missing")
        return

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        print(f"unreadable without pickle: {type(exc).__name__}: {exc}")
        return

    print(f"size_mb: {path.stat().st_size / (1024 * 1024):.2f}")
    print("keys:")

    id_like = []
    for k in data.files:
        arr = data[k]
        if arr.shape == ():
            try:
                value = arr.item()
            except Exception:
                value = "<scalar unreadable>"
            print(f"- {k}: scalar={value!r}, dtype={arr.dtype}")
        else:
            print(f"- {k}: shape={arr.shape}, dtype={arr.dtype}")

        low = k.lower()
        if any(s in low for s in ["traj", "trajectory", "seq", "sequence", "frame", "idx", "index", "path", "file", "scene", "env"]):
            id_like.append(k)

    print(f"id_like_keys: {id_like if id_like else 'NONE'}")


def audit_npz() -> None:
    print("\n# NPZ content audit")
    for path in NPZ_PATHS:
        describe_npz(path)


def audit_metrics() -> None:
    print("\n# Bestval metrics provenance")
    for p in sorted((ROOT / "outputs").glob("bestval_dinov2_cheap10_exp80_seed*/metrics.json")):
        print(f"\n## {p.relative_to(ROOT)}")
        try:
            obj = json.loads(p.read_text())
        except Exception as exc:
            print(f"unreadable: {exc}")
            continue

        for key in ["seed", "train", "val", "data", "best_policy", "best_error", "best_compute", "best_selected", "best_utility"]:
            if key in obj:
                print(f"- {key}: {obj[key]}")


def main() -> None:
    audit_code()
    audit_npz()
    audit_metrics()


if __name__ == "__main__":
    main()
