# Paper assets v29 capacity routing

## Scientific update

The capacity ladder shows non-monotonic OOD reliability: Medium beats Full on the hardest H=72 shift, while Tiny is often the best low-cost choice.
The learned multi-capacity router has stable positive gains on H=48, but not on H=36 or H=72. H=72 remains an oracle-headroom problem.

## Tables

- `reports/paper_assets_v29_capacity_routing/tables/table_v29_hard_ood_capacity_frontier.csv`
- `reports/paper_assets_v29_capacity_routing/tables/table_v29_hard_ood_capacity_frontier.tex`
- `reports/paper_assets_v29_capacity_routing/tables/table_v29_learned_router_bootstrap_best.csv`
- `reports/paper_assets_v29_capacity_routing/tables/table_v29_learned_router_bootstrap_best.tex`

## Figures

- `reports/paper_assets_v29_capacity_routing/figures/fig_v29_capacity_frontier_hard_ood.png`
- `reports/paper_assets_v29_capacity_routing/figures/fig_v29_learned_router_gain_bootstrap.png`

## Recommended claim

SPSM reveals non-monotonic capacity under controlled OOD interventions and shows that learned multi-capacity routing can provide statistically stable gains on specific hard shifts, while harder regimes expose remaining oracle headroom and the need for stronger routing signals.
