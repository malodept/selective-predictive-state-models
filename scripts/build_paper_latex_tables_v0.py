from pathlib import Path
import pandas as pd

ASSETS = Path("reports/paper_assets_v0/tables")
OUT = Path("paper/spsm_v0/tables")
OUT.mkdir(parents=True, exist_ok=True)

def esc(x):
    s = str(x)
    s = s.replace("_", r"\_")
    s = s.replace("±", r"$\pm$")
    return s

def tex_table(headers, rows, caption, label, out_name, small=True):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    if small:
        lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(esc(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(esc(v) for v in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    path = OUT / out_name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)

def f(x, n=3):
    return f"{float(x):.{n}f}"

def first_num(x):
    return str(x).split("±")[0].strip()

# Table 1 — OOD summary
ood = pd.read_csv(ASSETS / "table_v5_3_ood_variants.csv")
rows = []
for _, r in ood.iterrows():
    rows.append([
        r["variant"],
        first_num(r["moving-only strict top-1"]),
        first_num(r["same-action/diff-state"]),
        first_num(r["same-state/diff-action"]),
        first_num(r["strict @80% cov"]),
        first_num(r["strict @60% cov"]),
    ])

tex_table(
    ["Variant", "Top-1", "Same-action", "Same-state", "@80% cov.", "@60% cov."],
    rows,
    "Controlled OOD evaluation of the full state-action Transformer. The strongest degradation occurs under block3 shifts, especially on the same-action/different-state axis.",
    "tab:ood_summary",
    "table_ood_summary.tex",
)

# Table 2 — small full vs full
small = pd.read_csv(ASSETS / "table_v5_6_small_full_ood.csv")
rows = []
for _, r in small.iterrows():
    rows.append([
        r["variant"],
        first_num(r["full top-1 mean seeds 0-2"]),
        f(r["full seed0 top-1"], 3),
        f(r["small-full seed0 top-1"], 3),
        f(r["gap full seed0 - small"], 3),
    ])

tex_table(
    ["Variant", "Full mean", "Full seed0", "Small-full", "Gap"],
    rows,
    "Comparison between the reduced full model and the larger full model. The expensive model is not uniformly better, making value-of-computation non-trivial.",
    "tab:small_full",
    "table_small_full_summary.tex",
)

# Table 3 — routing at lambda = 0.10
ctx = pd.read_csv(ASSETS / "table_v5_8_context_gain_router.csv")
ctx010 = ctx[ctx["lambda"].round(2) == 0.10].copy()
ctx010 = ctx010[ctx010["variant"].isin(["block3_h36_seed13", "block3_h72_seed14"])]

rows = []
for _, r in ctx010.iterrows():
    rows.append([
        r["variant"],
        f(r["cheap util"], 3),
        f(r["full util"], 3),
        f(r["conf util"], 3),
        f(r["context-learned util"], 3),
        f(r["oracle util"], 3),
        f(r["context route"], 3),
    ])

tex_table(
    [r"Variant", r"Cheap", r"Full", r"Conf.", r"Context", r"Oracle", r"Route"],
    rows,
    r"Value-of-computation routing at $\lambda=0.10$. The context-aware router avoids harmful routing on block3\_h36 while routing more on block3\_h72, where the larger model helps.",
    "tab:routing_lambda010",
    "table_routing_lambda010.tex",
)

# Table 4 — routing at lambda = 0.30
ctx030 = ctx[ctx["lambda"].round(2) == 0.30].copy()
ctx030 = ctx030[ctx030["variant"].isin(["block3_h36_seed13", "block3_h72_seed14"])]

rows = []
for _, r in ctx030.iterrows():
    rows.append([
        r["variant"],
        f(r["cheap util"], 3),
        f(r["full util"], 3),
        f(r["conf util"], 3),
        f(r["context-learned util"], 3),
        f(r["oracle util"], 3),
        f(r["context route"], 3),
    ])

tex_table(
    [r"Variant", r"Cheap", r"Full", r"Conf.", r"Context", r"Oracle", r"Route"],
    rows,
    r"Value-of-computation routing at $\lambda=0.30$. At higher compute cost, the context-aware router remains conservative on block3\_h36 and selectively routes on block3\_h72.",
    "tab:routing_lambda030",
    "table_routing_lambda030.tex",
)
