from __future__ import annotations

from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v31_multiseed_capacity_ladder_ood")
OUT_PER_SEED = ROOT / "multiseed_capacity_ladder_per_seed.csv"
OUT_AGG = ROOT / "multiseed_capacity_ladder_aggregate.csv"
OUT_MD = ROOT / "multiseed_capacity_ladder_summary.md"

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

CAPACITIES = [
    ("tiny_full", "Tiny"),
    ("small_full", "Small"),
    ("medium_full", "Medium"),
    ("full", "Full"),
]

SEEDS = [0, 1, 2]


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


def fmt_mu_std(mu, sd):
    return f"{mu:.3f}±{sd:.3f}"


def main():
    rows = []

    for variant in VARIANTS:
        for cap_prefix, cap_label in CAPACITIES:
            for seed in SEEDS:
                model = f"{cap_prefix}_seed{seed}"
                vals = read_eval(variant, model)

                rows.append({
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "capacity": cap_prefix,
                    "capacity_label": cap_label,
                    "seed": seed,
                    "model": model,
                    "tie_aware_top1": vals.get("tie_aware_top1"),
                    "strict_top1": vals.get("strict_top1"),
                    "biased_argmin_top1": vals.get("biased_argmin_top1"),
                    "mean_tie_count": vals.get("mean_tie_count"),
                    "same_action_diff_state_win": vals.get("same_action_diff_state_win"),
                    "same_state_diff_action_win": vals.get("same_state_diff_action_win"),
                    "same_action_diff_state_margin": vals.get("same_action_diff_state_margin"),
                    "same_state_diff_action_margin": vals.get("same_state_diff_action_margin"),
                })

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT_PER_SEED, index=False)

    agg = (
        per_seed
        .groupby(["variant", "variant_label", "capacity", "capacity_label"], as_index=False)
        .agg(
            top1_mean=("tie_aware_top1", "mean"),
            top1_std=("tie_aware_top1", "std"),
            strict_mean=("strict_top1", "mean"),
            strict_std=("strict_top1", "std"),
            same_action_mean=("same_action_diff_state_win", "mean"),
            same_action_std=("same_action_diff_state_win", "std"),
            same_state_mean=("same_state_diff_action_win", "mean"),
            same_state_std=("same_state_diff_action_win", "std"),
            n=("seed", "count"),
        )
    )
    agg.to_csv(OUT_AGG, index=False)

    lines = []
    lines.append("# SPSM v31C multi-seed capacity ladder summary\n")
    lines.append("This repeats the OOD capacity ladder across Tiny/Small/Medium/Full with seeds 0, 1, and 2.")
    lines.append("The goal is to test whether the non-monotonic capacity result from v22/v29 is stable under retraining.\n")

    lines.append("## Multi-seed OOD top-1 by capacity\n")
    lines.append("| variant | Tiny | Small | Medium | Full | best mean | range of means |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")

    for variant in VARIANTS:
        sub = agg[agg["variant"] == variant].set_index("capacity_label")
        vals = {c: float(sub.loc[c, "top1_mean"]) for c in ["Tiny", "Small", "Medium", "Full"]}
        stds = {c: float(sub.loc[c, "top1_std"]) for c in ["Tiny", "Small", "Medium", "Full"]}
        best = max(vals, key=vals.get)
        spread = max(vals.values()) - min(vals.values())
        lines.append(
            f"| {VARIANT_LABELS[variant]} | "
            f"{fmt_mu_std(vals['Tiny'], stds['Tiny'])} | "
            f"{fmt_mu_std(vals['Small'], stds['Small'])} | "
            f"{fmt_mu_std(vals['Medium'], stds['Medium'])} | "
            f"{fmt_mu_std(vals['Full'], stds['Full'])} | "
            f"`{best}` | {spread:.6f} |"
        )

    lines.append("\n## Best capacity per individual seed\n")
    lines.append("| variant | seed0 | seed1 | seed2 |")
    lines.append("|---|---|---|---|")

    for variant in VARIANTS:
        sub = per_seed[per_seed["variant"] == variant]
        best_by_seed = []
        for seed in SEEDS:
            s = sub[sub["seed"] == seed]
            best = s.sort_values("tie_aware_top1", ascending=False).iloc[0]
            best_by_seed.append(f"`{best['capacity_label']}` ({best['tie_aware_top1']:.3f})")
        lines.append(f"| {VARIANT_LABELS[variant]} | {best_by_seed[0]} | {best_by_seed[1]} | {best_by_seed[2]} |")

    lines.append("\n## State/action decomposition, mean over seeds\n")
    lines.append("| variant | capacity | same-action/diff-state | same-state/diff-action |")
    lines.append("|---|---|---:|---:|")

    for variant in VARIANTS:
        sub = agg[agg["variant"] == variant]
        for _, r in sub.sort_values("capacity_label").iterrows():
            lines.append(
                f"| {r['variant_label']} | {r['capacity_label']} | "
                f"{r['same_action_mean']:.3f}±{r['same_action_std']:.3f} | "
                f"{r['same_state_mean']:.3f}±{r['same_state_std']:.3f} |"
            )

    lines.append("\n## Key robustness checks\n")

    h72 = agg[agg["variant"] == "block3_h72_seed14"].set_index("capacity_label")
    h48 = agg[agg["variant"] == "block3_h48_seed10"].set_index("capacity_label")
    h36 = agg[agg["variant"] == "block3_h36_seed13"].set_index("capacity_label")

    def delta(sub, a, b):
        return float(sub.loc[a, "top1_mean"] - sub.loc[b, "top1_mean"])

    lines.append(f"- H=72 Medium - Full mean gap: {delta(h72, 'Medium', 'Full'):.6f}.")
    lines.append(f"- H=72 Medium - Tiny mean gap: {delta(h72, 'Medium', 'Tiny'):.6f}.")
    lines.append(f"- H=48 Medium - Full mean gap: {delta(h48, 'Medium', 'Full'):.6f}.")
    lines.append(f"- H=48 Medium - Tiny mean gap: {delta(h48, 'Medium', 'Tiny'):.6f}.")
    lines.append(f"- H=36 Small - Full mean gap: {delta(h36, 'Small', 'Full'):.6f}.")
    lines.append(f"- H=36 Small - Tiny mean gap: {delta(h36, 'Small', 'Tiny'):.6f}.")

    lines.append("\n## Interpretation guide\n")
    lines.append("- If Medium remains above Full on H=72 and H=48 after averaging seeds, the non-monotonic capacity claim becomes much stronger.")
    lines.append("- If the best capacity changes across seeds, the paper should emphasize capacity-seed sensitivity rather than a single deterministic ranking.")
    lines.append("- If smaller capacities consistently beat Full on block3 variants, this supports the claim that larger predictors can be less reliable under constrained-geometry OOD.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(OUT_PER_SEED)
    print(OUT_AGG)
    print(OUT_MD)
    print()
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
