from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spsm.datasets.feature_npz import FeatureNPZPredictiveStateDataset
from spsm.datasets.synthetic import SyntheticDatasetConfig, SyntheticPredictiveStateDataset
from spsm.eval.calibration import expected_calibration_error
from spsm.eval.latency import count_parameters, profile_forward_latency
from spsm.eval.plotting import write_selector_csv
from spsm.eval.retrieval import retrieval_at_k
from spsm.eval.surprise import auprc_score, auroc_score, brier_score
from spsm.models.predictor import PredictiveStateModel
from spsm.models.selector import SelectorCosts, utility_for_policy
from spsm.training.checkpointing import save_checkpoint
from spsm.training.trainer import TrainConfig, Trainer
from spsm.utils.io import load_yaml, write_json
from spsm.utils.seed import seed_everything


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    z_preds, z_futures, clean_futures = [], [], []
    expected_labels, observed_labels, probs = [], [], []
    residuals_observed, residuals_clean = [], []
    for batch in loader:
        z_current = batch["z_current"].to(device)
        action = batch["action"].to(device)
        z_future = batch["z_future"].to(device)
        clean_future = batch.get("clean_future", z_future).to(device)
        expected_unreliable = batch.get("expected_unreliable", batch["is_surprise"]).to(device)
        observed_surprise = batch.get("observed_surprise", batch["is_surprise"]).to(device)

        out = model(z_current, action)
        prob = torch.sigmoid(out["reliability_logit"])
        residual_observed = torch.mean((out["z_pred"] - z_future) ** 2, dim=-1)
        residual_clean = torch.mean((out["z_pred"] - clean_future) ** 2, dim=-1)

        z_preds.append(out["z_pred"].detach().cpu())
        z_futures.append(z_future.detach().cpu())
        clean_futures.append(clean_future.detach().cpu())
        expected_labels.append(expected_unreliable.detach().cpu())
        observed_labels.append(observed_surprise.detach().cpu())
        probs.append(prob.detach().cpu())
        residuals_observed.append(residual_observed.detach().cpu())
        residuals_clean.append(residual_clean.detach().cpu())
    return {
        "z_pred": torch.cat(z_preds),
        "z_future": torch.cat(z_futures),
        "clean_future": torch.cat(clean_futures),
        "expected_labels": torch.cat(expected_labels),
        "observed_labels": torch.cat(observed_labels),
        "probs": torch.cat(probs),
        "residuals_observed": torch.cat(residuals_observed),
        "residuals_clean": torch.cat(residuals_clean),
    }


def build_dataset(cfg: dict, split: str):
    data_cfg = dict(cfg["data"])
    name = data_cfg.pop("name")
    if name == "synthetic":
        n = data_cfg.pop("n_train" if split == "train" else "n_val")
        data_cfg.pop("n_val" if split == "train" else "n_train", None)
        if split == "val":
            data_cfg["seed"] = int(data_cfg.get("seed", 0)) + 1000
        return SyntheticPredictiveStateDataset(SyntheticDatasetConfig(n_samples=n, **data_cfg))
    if name == "feature_npz":
        path = data_cfg["train_path" if split == "train" else "val_path"]
        return FeatureNPZPredictiveStateDataset(path)
    raise ValueError(f"Unknown dataset name: {name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    out_dir = Path(cfg.get("output_dir", "outputs/week1_mre"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_cfg = TrainConfig(**cfg["training"])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)

    model_cfg = cfg["model"]
    model = PredictiveStateModel(
        latent_dim=model_cfg["latent_dim"],
        action_dim=model_cfg.get("action_dim", 0),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        predictor_layers=model_cfg.get("predictor_layers", 2),
        reliability_hidden_dim=model_cfg.get("reliability_hidden_dim", 128),
        dropout=model_cfg.get("dropout", 0.0),
    )
    device = torch.device(args.device)
    trainer = Trainer(model, train_cfg, device)
    history = trainer.fit(train_loader)

    outputs = collect_outputs(model, val_loader, device)
    valid = outputs["observed_labels"] == 0
    retrieval = retrieval_at_k(
        outputs["z_pred"][valid], outputs["clean_future"][valid], cfg["eval"]["retrieval_ks"]
    )
    probs = outputs["probs"]
    expected_labels = outputs["expected_labels"]
    observed_labels = outputs["observed_labels"]
    residuals_observed = outputs["residuals_observed"]
    residuals_clean = outputs["residuals_clean"]

    # Save per-sample arrays for diagnostic plots and calibration analysis.
    np.savez(
        out_dir / "eval_arrays.npz",
        probs=probs.numpy(),
        residuals_observed=residuals_observed.numpy(),
        residuals_clean=residuals_clean.numpy(),
        expected_labels=expected_labels.numpy(),
        observed_labels=observed_labels.numpy(),
    )

    surprise_metrics = {
        "expected_learned_auroc": auroc_score(expected_labels, probs),
        "expected_learned_auprc": auprc_score(expected_labels, probs),
        "expected_learned_brier": brier_score(expected_labels, probs),
        "expected_learned_ece": expected_calibration_error(expected_labels, probs),
        "expected_residual_auroc": auroc_score(expected_labels, residuals_clean),
        "expected_residual_auprc": auprc_score(expected_labels, residuals_clean),
        "observed_residual_auroc": auroc_score(observed_labels, residuals_observed),
        "observed_residual_auprc": auprc_score(observed_labels, residuals_observed),
        "observed_learned_auroc": auroc_score(observed_labels, probs),
        "observed_learned_auprc": auprc_score(observed_labels, probs),
    }

    thresholds = cfg["eval"]["selector_thresholds"]
    costs = SelectorCosts(
        cheap_compute=float(cfg["eval"]["cheap_compute"]),
        optional_compute=float(cfg["eval"]["optional_compute"]),
        utility_scale=float(cfg["eval"].get("utility_scale", 1.0)),
        compute_penalty=float(cfg["eval"].get("compute_penalty", 0.15)),
        hard_refinement_factor=float(cfg["eval"].get("hard_refinement_factor", 0.25)),
        easy_refinement_factor=float(cfg["eval"].get("easy_refinement_factor", 0.95)),
    )
    # Selector uses pre-observation learned reliability; utility is measured on clean prediction error.
    # The optional expert is useful mainly on expected-unreliable samples.
    selector_rows = []
    for t in thresholds:
        row = utility_for_policy(residuals_clean, probs, t, costs, expected_labels)
        row["policy"] = f"threshold={t:.2f}"
        selector_rows.append(row)

    # Static references. These are policies, not thresholds.
    cheap_row = utility_for_policy(
        residuals_clean, torch.zeros_like(probs), 1.0, costs, expected_labels
    )
    cheap_row["policy"] = "cheap-only"
    cheap_row["threshold"] = None
    selector_rows.append(cheap_row)

    expensive_row = utility_for_policy(
        residuals_clean, torch.ones_like(probs), 0.0, costs, expected_labels
    )
    expensive_row["policy"] = "all-expensive"
    expensive_row["threshold"] = None
    selector_rows.append(expensive_row)
    write_selector_csv(selector_rows, out_dir / "selector_utility.csv")

    sample = next(iter(val_loader))
    latency = profile_forward_latency(
        model.to(device),
        sample["z_current"].to(device),
        sample["action"].to(device),
        warmup=int(cfg["eval"].get("latency_warmup", 2)),
        repeats=int(cfg["eval"].get("latency_repeats", 5)),
    )

    metrics = {
        "history": history,
        "retrieval": retrieval,
        "surprise": surprise_metrics,
        "selector_best_utility": max(row["utility"] for row in selector_rows),
        "selector_rows": selector_rows,
        "parameter_count": count_parameters(model),
        "latency": latency,
    }
    write_json(metrics, out_dir / "metrics.json")
    save_checkpoint(model, out_dir / "checkpoint.pt", extra={"config": cfg, "metrics": metrics})
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"Retrieval: {retrieval}")
    print(f"Surprise: {surprise_metrics}")


if __name__ == "__main__":
    main()
