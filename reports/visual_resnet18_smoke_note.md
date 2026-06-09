# ResNet18 visual-feature smoke test

This experiment replaces the dependency-light patch-mean encoder with a frozen ResNet18 visual feature extractor. The purpose is not to claim a final visual world model, but to verify that the SPSM pipeline can operate on features produced by a real visual backbone.

Pipeline:

1. Generate or provide short image sequences.
2. Encode each frame with frozen ResNet18.
3. Build transitions `(z_t, a_t, z_{t+Δ})`, where `a_t` is currently the normalized temporal gap.
4. Train the same predictive-state model, reliability head, and selective-compute policy.

This is the bridge between the synthetic latent proof of concept and future public datasets such as TartanAir, TartanDrive, DROID, or other trajectory datasets.
