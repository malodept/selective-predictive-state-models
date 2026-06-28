from __future__ import annotations

from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v22_capacity_ladder_ood")
OUT_CSV = ROOT / "capacity_ladder_ood_summary.csv"
OUT_MD = ROOT / "capacity_ladder_ood_summary.md"

CKPT_ROOT = Path("outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer")

VARIANTS = [
    "block2_h72_seed11",
    "block2_v18_seed12",
    "block3_h36_seed13",
    "block3_h48_seed10",
    "block3_h72_seed14",
]

VARIANT_LABELS = {
    "block2_h72_seed11": "2-block, H=72",
    "block2_v18_seed12": "2-block, V=1.8",
    "block3_h36_seed13": "3-block, H=36",
    "block3_h48_seed10": "3-block, H=48",
    "block3_h72_seed14": "3-block, H=72",
}

MODELS = [
    ("tiny_full_seed0", "Tiny", 64, 1),
    ("small_full_seed0", "Small", 128, 1),
    ("medium_full_seed0", "Medium", 192, 2),
    ("full_seed0", "Full", None, None),
]


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_metric(obj, names):
    wanted = {norm_key(x) for x in names}

    if isinstance(obj, dict):
        for k, v in obj.items():
            if norm_key(k) in wanted and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            found = find_metric(v, names)
            if found is not None:
                return found

    if isinstance(obj, list):
        for v in obj:
            found = find_metric(v, names)
            if found is not None:
                return found

    return None


def parse_md_fallback(md_path: Path) -> dict:
    out = {}
    if not md_path.exists():
        return out

    txt = md_path.read_text(encoding="utf-8", errors="ignore")

    metric_map = {
        "biased argmin top-1": "biased_argmin_top1",
        "strict top-1": "strict_top1",
        "tie-aware top-1": "tie_aware_top1",
        "mean tie count at minimum": "mean_tie_count",
    }

    for line in txt.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([-+0-9.eE]+)\s*\|", line)
        if m:
            key = m.group(1).strip()
            val = float(m.group(2))
            if key in metric_map:
                out[metric_map[key]] = val

        if "same_action_diff_state" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 7:
                out["same_action_diff_state_win"] = float(parts[1])
                out["same_action_diff_state_margin"] = float(parts[5])
                out["same_action_diff_state_count"] = int(float(parts[6]))

        if "same_state_diff_action" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 7:
                out["same_state_diff_action_win"] = float(parts[1])
                out["same_state_diff_action_margin"] = float(parts[5])
                out["same_state_diff_action_count"] = int(float(parts[6]))

    return out


def read_eval(variant: str, model: str) -> dict:
    jpath = ROOT / variant / f"moving_only_{model}.json"
    md_path = ROOT / variant / f"moving_only_{model}.md"

    vals = parse_md_fallback(md_path)

    if jpath.exists():
        try:
            obj = json.loads(jpath.read_text())
            vals.setdefault("tie_aware_top1", find_metric(obj, ["tie_aware_top1", "tie-aware top-1", "tie aware top 1"]))
            vals.setdefault("strict_top1", find_metric(obj, ["strict_top1", "strict top-1", "strict top 1"]))
            vals.setdefault("biased_argmin_top1", find_metric(obj, ["biased_argmin_top1", "biased argmin top-1"]))
            vals.setdefault("mean_tie_count", find_metric(obj, ["mean_tie_count", "mean tie count at minimum"]))
        except Exception as e:
            vals["json_error"] = str(e)

    return vals


def main():
    rows = []

    for variant in VARIANTS:
        for model, label, dim, layers in MODELS:
            vals = read_eval(variant, model)

            ckpt = CKPT_ROOT / model / "checkpoint.pt"
            size_mb = ckpt.stat().st_size / 1024 / 1024 if ckpt.exists() else None

            rows.append({
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "model": model,
                "model_label": label,
                "model_dim": dim,
                "layers": layers,
                "checkpoint_mb": size_mb,
                "tie_aware_top1": vals.get("tie_aware_top1"),
                "strict_top1": vals.get("strict_top1"),
                "biased_argmin_top1": vals.get("biased_argmin_top1"),
                "mean_tie_count": vals.get("mean_tie_count"),
                "same_action_diff_state_win": vals.get("same_action_diff_state_win"),
                "same_state_diff_action_win": vals.get("same_state_diff_action_win"),
                "same_action_diff_state_margin": vals.get("same_action_diff_state_margin"),
                "same_state_diff_action_margin": vals.get("same_state_diff_action_margin"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v22 OOD capacity ladder\n")
    lines.append("This evaluates the same mixed-hard moving-only OOD protocol across four predictor capacities: tiny, small, medium, and full.")
    lines.append("ID performance is saturated for all capacities, so this table focuses on OOD reliability.\n")

    lines.append("## Top-1 by capacity and OOD variant\n")
    lines.append("| variant | tiny | small | medium | full | best | range |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")

    for variant in VARIANTS:
        sub = df[df["variant"] == variant].set_index("model")
        vals = {
            model: float(sub.loc[model, "tie_aware_top1"])
            for model, _, _, _ in MODELS
        }
        best_model = max(vals, key=vals.get)
        label_map = {m: lab for m, lab, _, _ in MODELS}
        spread = max(vals.values()) - min(vals.values())

        lines.append(
            f"| {VARIANT_LABELS[variant]} | "
            f"{vals['tiny_full_seed0']:.6f} | "
            f"{vals['small_full_seed0']:.6f} | "
            f"{vals['medium_full_seed0']:.6f} | "
            f"{vals['full_seed0']:.6f} | "
            f"`{label_map[best_model]}` | "
            f"{spread:.6f} |"
        )

    lines.append("\n## State/action decomposition\n")
    lines.append("| variant | model | same-action/diff-state win | same-state/diff-action win |")
    lines.append("|---|---|---:|---:|")

    for variant in VARIANTS:
        sub = df[df["variant"] == variant]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['variant_label']} | {r['model_label']} | "
                f"{r['same_action_diff_state_win']:.6f} | "
                f"{r['same_state_diff_action_win']:.6f} |"
            )

    lines.append("\n## Checkpoint size\n")
    lines.append("| model | checkpoint MB |")
    lines.append("|---|---:|")

    for model, label, _, _ in MODELS:
        ckpt = CKPT_ROOT / model / "checkpoint.pt"
        size = ckpt.stat().st_size / 1024 / 1024 if ckpt.exists() else float("nan")
        lines.append(f"| {label} | {size:.2f} |")

    lines.append("\n## Interpretation guide\n")
    lines.append("- If larger models are not consistently better, this supports the non-uniform cheap/full ordering observed in VoC.")
    lines.append("- If `medium` often matches or beats `full`, the expensive endpoint is not the right upper model.")
    lines.append("- If the spread grows on block3 shifts, capacity matters specifically under harder OOD.")
    lines.append("- The next step is a cost-normalized capacity VoC table, not another router family.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
