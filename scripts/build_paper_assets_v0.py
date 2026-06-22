from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder")
OUT = Path("reports/paper_assets_v0")
TABLES = OUT / "tables"
FIGS = OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

ORDER = [
    "id_block2_h36_seed0",
    "block2_h72_seed11",
    "block2_v18_seed12",
    "block3_h36_seed13",
    "block3_h48_seed10",
    "block3_h72_seed14",
]

OOD_ORDER = [
    "block2_h72_seed11",
    "block2_v18_seed12",
    "block3_h36_seed13",
    "block3_h48_seed10",
    "block3_h72_seed14",
]

def parse_md_tables(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    tables = []
    current = []

    for line in text.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            current.append(line.strip())
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)

    dfs = []
    for table in tables:
        rows = []
        for line in table:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                continue
            rows.append(cells)

        if len(rows) >= 2:
            header = rows[0]
            data = rows[1:]
            width = len(header)
            data = [r for r in data if len(r) == width]
            dfs.append(pd.DataFrame(data, columns=header))

    return dfs

def first_float(x):
    if pd.isna(x):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else None

def as_float_series(s):
    return s.map(first_float).astype(float)

def save_df(df, name):
    path = TABLES / name
    df.to_csv(path, index=False)
    return path

def plot_bar(df, x_col, y_col, title, ylabel, out_name):
    fig = plt.figure(figsize=(10, 5))
    plt.bar(df[x_col], df[y_col])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out = FIGS / out_name
    plt.savefig(out, dpi=200)
    plt.close(fig)
    return out

def plot_lines(df, x_col, y_cols, title, ylabel, out_name):
    fig = plt.figure(figsize=(8, 5))
    for col in y_cols:
        plt.plot(df[x_col], df[col], marker="o", label=col)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out = FIGS / out_name
    plt.savefig(out, dpi=200)
    plt.close(fig)
    return out

def plot_grouped_small_vs_full(df):
    x = list(range(len(df)))
    width = 0.35

    fig = plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], df["full seed0 top-1"], width, label="full seed0")
    plt.bar([i + width / 2 for i in x], df["small-full seed0 top-1"], width, label="small-full seed0")
    plt.xticks(x, df["variant"], rotation=30, ha="right")
    plt.ylabel("moving-only strict top-1")
    plt.title("Small-full vs full under controlled OOD shifts")
    plt.legend()
    plt.tight_layout()
    out = FIGS / "small_full_vs_full_ood.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    return out

def main():
    outputs = []

    # v5.3 OOD summary
    ood_path = ROOT / "v5_3_ood" / "v5_3_ood_variants_summary.md"
    ood = parse_md_tables(ood_path)[0]
    ood["moving_top1_mean"] = as_float_series(ood["moving-only strict top-1"])
    ood["same_action_diff_state_mean"] = as_float_series(ood["same-action/diff-state"])
    ood["same_state_diff_action_mean"] = as_float_series(ood["same-state/diff-action"])
    ood["variant"] = pd.Categorical(ood["variant"], categories=OOD_ORDER, ordered=True)
    ood = ood.sort_values("variant")
    outputs.append(save_df(ood, "table_v5_3_ood_variants.csv"))

    outputs.append(plot_bar(
        ood,
        "variant",
        "moving_top1_mean",
        "Full model OOD performance by controlled shift",
        "moving-only strict top-1",
        "fig_v5_3_ood_top1_by_variant.png",
    ))

    # v5.6 small-full OOD summary
    small_path = ROOT / "v5_6_small_full" / "small_full_ood_summary.md"
    small = parse_md_tables(small_path)[0]
    small["full_top1_mean"] = as_float_series(small["full top-1 mean seeds 0-2"])
    small["full seed0 top-1"] = as_float_series(small["full seed0 top-1"])
    small["small-full seed0 top-1"] = as_float_series(small["small-full seed0 top-1"])
    small["gap full seed0 - small"] = as_float_series(small["gap full seed0 - small"])
    small["variant"] = pd.Categorical(small["variant"], categories=OOD_ORDER, ordered=True)
    small = small.sort_values("variant")
    outputs.append(save_df(small, "table_v5_6_small_full_ood.csv"))
    outputs.append(plot_grouped_small_vs_full(small))

    outputs.append(plot_bar(
        small,
        "variant",
        "gap full seed0 - small",
        "Full minus small-full OOD gap",
        "top-1 gap",
        "fig_v5_6_full_minus_small_gap.png",
    ))

    # v5.8 context router summary
    ctx_path = ROOT / "v5_8_context_gain_router" / "context_gain_router_summary.md"
    ctx = parse_md_tables(ctx_path)[0]
    for col in [
        "lambda",
        "cheap util",
        "full util",
        "oracle util",
        "oracle route",
        "conf util",
        "conf route",
        "context-learned util",
        "context route",
        "context acc",
    ]:
        ctx[col] = as_float_series(ctx[col])
    outputs.append(save_df(ctx, "table_v5_8_context_gain_router.csv"))

    # Utility vs lambda for the two key hard-OOD variants
    for variant in ["block3_h36_seed13", "block3_h72_seed14"]:
        sub = ctx[ctx["variant"] == variant].copy().sort_values("lambda")
        sub = sub.rename(columns={
            "cheap util": "cheap-only",
            "full util": "full-only",
            "conf util": "confidence",
            "context-learned util": "context-aware",
            "oracle util": "oracle",
        })
        outputs.append(plot_lines(
            sub,
            "lambda",
            ["cheap-only", "full-only", "confidence", "context-aware", "oracle"],
            f"Value-of-computation utility vs lambda — {variant}",
            "utility",
            f"fig_v5_8_utility_vs_lambda_{variant}.png",
        ))

    # Route rate vs lambda for key variants
    for variant in ["block3_h36_seed13", "block3_h72_seed14"]:
        sub = ctx[ctx["variant"] == variant].copy().sort_values("lambda")
        sub = sub.rename(columns={
            "conf route": "confidence route rate",
            "context route": "context route rate",
            "oracle route": "oracle route rate",
        })
        outputs.append(plot_lines(
            sub,
            "lambda",
            ["confidence route rate", "context route rate", "oracle route rate"],
            f"Routing rate vs lambda — {variant}",
            "route rate",
            f"fig_v5_8_route_rate_vs_lambda_{variant}.png",
        ))

    manifest = OUT / "paper_assets_manifest.md"
    lines = ["# Paper assets v0\n"]
    lines.append("Generated from existing SPSM v5.3, v5.6, and v5.8 results.\n")
    lines.append("## Tables\n")
    for p in sorted(TABLES.glob("*")):
        lines.append(f"- `{p}`")
    lines.append("\n## Figures\n")
    for p in sorted(FIGS.glob("*")):
        lines.append(f"- `{p}`")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs.append(manifest)

    print(manifest.read_text())
    print("\nGenerated:")
    for p in outputs:
        print(p)

if __name__ == "__main__":
    main()
