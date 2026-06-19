# PyBullet obstacle block-effect audit

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- samples: `2500`
- z shape: `(2500, 16, 384)`
- blocked fraction: `0.400000`

| action | blocked | count | dx mean | dy mean | disp norm mean | latent delta norm mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stay | unblocked | 500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| stay | blocked | 0 | NA | NA | NA | NA |
| right | unblocked | 248 | 0.499024 | 0.000000 | 0.499024 | 22.398773 |
| right | blocked | 252 | 0.180010 | 0.000000 | 0.180010 | 17.736547 |
| left | unblocked | 260 | -0.499024 | 0.000000 | 0.499024 | 22.925124 |
| left | blocked | 240 | -0.180010 | 0.000000 | 0.180010 | 17.557243 |
| forward | unblocked | 255 | 0.000000 | 0.499024 | 0.499024 | 25.363345 |
| forward | blocked | 245 | 0.000000 | 0.180010 | 0.180010 | 18.082665 |
| backward | unblocked | 237 | 0.000000 | -0.499024 | 0.499024 | 25.142088 |
| backward | blocked | 263 | 0.000000 | -0.180010 | 0.180010 | 19.269951 |

## Interpretation

If blocked and unblocked versions of the same action have similar physical and latent displacement statistics, the obstacle protocol is too weak.
If physical displacement changes but latent displacement remains action-template-like, the representation suppresses the state-dependent collision detail.
