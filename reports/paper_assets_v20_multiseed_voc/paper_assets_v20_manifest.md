# Paper assets v20 multi-seed VoC

## Main scientific update

The single-seed v15 result becomes a multi-cheap-seed stability result.
On 3-block H=72, routing remains useful, but the best router varies across cheap-model seeds.
Latent-state features provide a positive mean advantage at several compute costs, especially for KNN at λ=0.05–0.20, but the effect is small relative to cheap-seed variability.

## Tables

- `reports/paper_assets_v20_multiseed_voc/tables/table_v20_h72_multiseed_locked_voc.csv`
- `reports/paper_assets_v20_multiseed_voc/tables/table_v20_h72_multiseed_locked_voc.tex`
- `reports/paper_assets_v20_multiseed_voc/tables/table_v20_latent_advantage_h72.csv`

## Figures

- `reports/paper_assets_v20_multiseed_voc/figures/fig_v20_latent_advantage_h72.png`
- `reports/paper_assets_v20_multiseed_voc/figures/fig_v20_multiseed_utility_h72.png`

## Recommended paper claim

Use: observable latent-state features give directionally positive routing gains on the hardest long-horizon shift, but multi-seed results show that cheap-model realization is a major factor. The strongest robust contribution is therefore value-of-computation routing under non-uniform cheap/full model ordering, with latent-state features as a promising stabilizing signal.
