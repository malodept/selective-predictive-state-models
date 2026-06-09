# Datasets

## Synthetic latent sequences

Used only for repository smoke tests and early metric validation. This dataset should never be presented as a final scientific result.

## Public controlled trajectory data

Planned first real benchmark. The goal is to create short clips, extract frozen teacher features, and train a next-latent predictor.

## Real transfer data

Planned second-stage benchmark for robustness, reliability, and utility-vs-compute generalization.

## Real-video smoke input

For the first real-observation smoke test, place local videos in `data/raw_videos/`. The helper script
`scripts/extract_frames_from_videos.py` extracts fixed-rate frame sequences into `data/real_video_sequences/`.
These sequences can then be processed by `scripts/extract_image_sequence_features.py` using a frozen visual encoder.
