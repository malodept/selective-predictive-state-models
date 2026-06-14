from pathlib import Path

rows = [
    # lambda, selected policy, selected test utility, selected test gain, selected fraction
    (0.0000, "action_norm router", 0.016816, 0.016816, 0.947489),
    (0.0025, "action_norm router", 0.014284, 0.016581, 0.918811),
    (0.0050, "action_norm router", 0.011816, 0.016267, 0.890192),
    (0.0100, "action_norm router", 0.007235, 0.015078, 0.784288),
    (0.0150, "action_norm router", 0.003313, 0.015078, 0.784288),
    (0.0200, "action_norm router", 0.001713, 0.008167, 0.322676),
    (0.0300, "action_norm router", -0.000542, 0.005468, 0.200353),
]

always_gain = 0.017418

lines = [
    "# Directional selective utility compact summary",
    "",
    "Utility is defined as `gain - lambda * selected_fraction`.",
    "",
    "| lambda | selected policy | selected utility | always-residual utility | identity utility | selected gain | selected fraction | best test policy among three |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
]

for lam, policy, util, gain, frac in rows:
    always_util = always_gain - lam
    identity_util = 0.0

    candidates = [
        ("selected", util),
        ("always_residual", always_util),
        ("identity_only", identity_util),
    ]
    best = max(candidates, key=lambda x: x[1])[0]

    lines.append(
        f"| {lam:.4f} | {policy} | {util:.6f} | {always_util:.6f} | "
        f"{identity_util:.6f} | {gain:.6f} | {frac:.6f} | {best} |"
    )

out = Path("reports/tables/protocol/directional_selective_utility/directional_selective_utility_compact.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")

print(out)
print(out.read_text())
