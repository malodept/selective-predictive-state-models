from pathlib import Path
import json
import math


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pm(xs):
    return f"{mean(xs):.6f} ± {std(xs):.6f}"


configs = {
    "V-JEPA 2.1 action": [
        Path(f"outputs/envsplit_vjepa2_1_base384_causal16_transformer_mse_seed{s}/metrics.json")
        for s in [0, 1, 2]
    ],
    "V-JEPA 2.1 no-action": [
        Path(f"outputs/envsplit_vjepa2_1_base384_causal16_transformer_noaction_mse_seed{s}/metrics.json")
        for s in [0, 1, 2]
    ],
}

out_dir = Path("reports/tables/protocol/vjepa2_1_action_grounding")
out_dir.mkdir(parents=True, exist_ok=True)

lines = [
    "# V-JEPA 2.1 action vs no-action multiseed summary",
    "",
    "| model | seeds | identity error | raw error | global alpha | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | pred Δ norm med | best epoch |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

loaded = {}

for name, paths in configs.items():
    rows = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        m = json.loads(p.read_text())
        r = m["test"]
        rows.append({
            "identity": r["identity_error"],
            "raw": r["raw_error"],
            "alpha": r["global_alpha"],
            "global": r["global_error"],
            "gain": r["global_improvement_vs_identity"],
            "relative": 100.0 * r["global_improvement_vs_identity"] / r["identity_error"],
            "cosine": r["cosine_mean"],
            "positive": r["cosine_positive_frac"],
            "pred_norm": r["pred_delta_norm_median"],
            "best_epoch": float(m.get("best_epoch", 0)),
        })

    loaded[name] = rows

    lines.append(
        f"| {name} | {len(rows)} | "
        f"{pm([x['identity'] for x in rows])} | "
        f"{pm([x['raw'] for x in rows])} | "
        f"{pm([x['alpha'] for x in rows])} | "
        f"{pm([x['global'] for x in rows])} | "
        f"{pm([x['gain'] for x in rows])} | "
        f"{pm([x['relative'] for x in rows])}% | "
        f"{pm([x['cosine'] for x in rows])} | "
        f"{pm([x['positive'] for x in rows])} | "
        f"{pm([x['pred_norm'] for x in rows])} | "
        f"{pm([x['best_epoch'] for x in rows])} |"
    )

action_gain = mean([x["gain"] for x in loaded["V-JEPA 2.1 action"]])
noaction_gain = mean([x["gain"] for x in loaded["V-JEPA 2.1 no-action"]])
action_error = mean([x["global"] for x in loaded["V-JEPA 2.1 action"]])
noaction_error = mean([x["global"] for x in loaded["V-JEPA 2.1 no-action"]])

lines += [
    "",
    "## Interpretation",
    "",
    f"- Mean action gain: `{action_gain:.6f}`.",
    f"- Mean no-action gain: `{noaction_gain:.6f}`.",
    f"- Mean action global error: `{action_error:.6f}`.",
    f"- Mean no-action global error: `{noaction_error:.6f}`.",
    f"- Action minus no-action gain: `{action_gain - noaction_gain:.6f}`.",
    f"- No-action error minus action error: `{noaction_error - action_error:.6f}`.",
    "",
    "This table tests whether V-JEPA 2.1 makes action conditioning useful beyond latent video flow.",
]

out = out_dir / "vjepa2_1_action_vs_noaction_multiseed_summary.md"
out.write_text("\n".join(lines) + "\n")
print(out)
print(out.read_text())
