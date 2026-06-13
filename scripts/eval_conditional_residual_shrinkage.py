from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_envsplit_residual_predictor import MLP, load_npz, feature_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", choices=["cheap_residual", "expensive_residual"], default="cheap_residual")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def mse_per_sample(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean((a - b) ** 2, axis=1)


def mean_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(mse_per_sample(a, b).mean())


def torch_stats_to_numpy(stats: dict) -> dict[str, np.ndarray]:
    out = {}
    for k, v in stats.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().numpy().astype(np.float32)
        else:
            out[k] = np.asarray(v, dtype=np.float32)
    return out


def normalized_input(data: dict[str, np.ndarray], stats: dict[str, np.ndarray]) -> np.ndarray:
    x = feature_matrix(data).astype(np.float32)
    return ((x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)


def build_model(ckpt: dict, model_name: str, device: str) -> MLP:
    cfg = ckpt["metrics"]["model"]

    if model_name == "cheap_residual":
        state_key = "cheap"
        hidden = int(cfg["cheap_hidden"])
        layers = int(cfg["cheap_layers"])
    else:
        state_key = "expensive"
        hidden = int(cfg["expensive_hidden"])
        layers = int(cfg["expensive_layers"])

    model = MLP(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["latent_dim"]),
        hidden_dim=hidden,
        layers=layers,
        dropout=float(cfg.get("dropout", 0.0)),
        zero_init_final=True,
    ).to(device)

    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model


@torch.no_grad()
def predict_delta(model: torch.nn.Module, x: np.ndarray, stats: dict[str, np.ndarray], batch_size: int, device: str) -> np.ndarray:
    outs = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)

    for xb in loader:
        xb = xb.to(device)
        pred_norm = model(xb).detach().cpu().numpy().astype(np.float32)
        outs.append(pred_norm)

    pred_norm = np.concatenate(outs, axis=0)
    delta = pred_norm * stats["delta_std"][None, :] + stats["delta_mean"][None, :]
    return delta.astype(np.float32)


def oracle_alpha(delta_hat: np.ndarray, true_delta: np.ndarray) -> np.ndarray:
    num = np.sum(delta_hat * true_delta, axis=1)
    den = np.sum(delta_hat * delta_hat, axis=1) + 1e-12
    alpha = num / den
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def compact_alpha_features(data: dict[str, np.ndarray], delta_hat: np.ndarray) -> np.ndarray:
    z = data["z_current"].astype(np.float32)
    action = data["action"].astype(np.float32)

    parts = [action]

    if "gap" in data:
        parts.append(data["gap"].astype(np.float32)[:, None])

    trans_norm = np.linalg.norm(action[:, : min(3, action.shape[1])], axis=1, keepdims=True)
    action_norm = np.linalg.norm(action, axis=1, keepdims=True)
    delta_norm = np.linalg.norm(delta_hat, axis=1, keepdims=True)
    z_norm = np.linalg.norm(z, axis=1, keepdims=True)
    pred_norm = np.linalg.norm(z + delta_hat, axis=1, keepdims=True)

    cos_z_delta = np.sum(z * delta_hat, axis=1, keepdims=True) / (
        np.linalg.norm(z, axis=1, keepdims=True) * np.linalg.norm(delta_hat, axis=1, keepdims=True) + 1e-12
    )

    parts += [trans_norm, action_norm, delta_norm, z_norm, pred_norm, cos_z_delta]

    x = np.concatenate(parts, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def select_global_alpha(z0: np.ndarray, y: np.ndarray, delta_hat: np.ndarray) -> tuple[float, float]:
    best_alpha = 0.0
    best_error = mean_mse(z0, y)

    for alpha in np.linspace(0.0, 1.0, 101):
        err = mean_mse(z0 + alpha * delta_hat, y)
        if err < best_error:
            best_error = err
            best_alpha = float(alpha)

    return best_alpha, best_error


def select_mixing_gamma(
    z0: np.ndarray,
    y: np.ndarray,
    delta_hat: np.ndarray,
    alpha_global: float,
    alpha_cond: np.ndarray,
) -> tuple[float, float]:
    best_gamma = 0.0
    best_error = mean_mse(z0 + alpha_global * delta_hat, y)

    for gamma in np.linspace(0.0, 1.0, 101):
        alpha = np.clip((1.0 - gamma) * alpha_global + gamma * alpha_cond, 0.0, 1.0)
        err = mean_mse(z0 + alpha[:, None] * delta_hat, y)
        if err < best_error:
            best_error = err
            best_gamma = float(gamma)

    return best_gamma, best_error


def summarize_split(
    split_name: str,
    data: dict[str, np.ndarray],
    delta_hat: np.ndarray,
    alpha_global: float,
    alpha_cond_raw: np.ndarray,
    gamma: float,
) -> list[dict]:
    z0 = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)
    true_delta = y - z0

    identity_error = mean_mse(z0, y)
    raw_error = mean_mse(z0 + delta_hat, y)

    global_pred = z0 + alpha_global * delta_hat
    global_error = mean_mse(global_pred, y)

    cond_raw = np.clip(alpha_cond_raw, 0.0, 1.0).astype(np.float32)
    cond_raw_error = mean_mse(z0 + cond_raw[:, None] * delta_hat, y)

    cond_mix = np.clip((1.0 - gamma) * alpha_global + gamma * cond_raw, 0.0, 1.0).astype(np.float32)
    cond_mix_error = mean_mse(z0 + cond_mix[:, None] * delta_hat, y)

    alpha_oracle = oracle_alpha(delta_hat, true_delta)
    oracle_error = mean_mse(z0 + alpha_oracle[:, None] * delta_hat, y)

    rows = []
    for method, err, alpha_mean, alpha_std in [
        ("identity", identity_error, 0.0, 0.0),
        ("raw_residual_alpha_1", raw_error, 1.0, 0.0),
        ("global_alpha_val", global_error, alpha_global, 0.0),
        ("conditional_alpha_raw", cond_raw_error, float(cond_raw.mean()), float(cond_raw.std())),
        ("conditional_alpha_mixed_with_global", cond_mix_error, float(cond_mix.mean()), float(cond_mix.std())),
        ("oracle_per_sample_alpha", oracle_error, float(alpha_oracle.mean()), float(alpha_oracle.std())),
    ]:
        rows.append(
            {
                "split": split_name,
                "method": method,
                "error": err,
                "identity_error": identity_error,
                "improvement_vs_identity": identity_error - err,
                "ratio_vs_identity": err / identity_error,
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x: float) -> str:
    return f"{x:.6f}"


def write_md(path: Path, rows: list[dict], alpha_global: float, gamma: float) -> None:
    lines = [
        "# Conditional residual shrinkage diagnostic",
        "",
        "Prediction family:",
        "",
        "`z_pred = z_current + alpha(x) * delta_hat`",
        "",
        f"Global alpha selected on validation: `{alpha_global:.4f}`.",
        f"Mixing coefficient selected on validation: `gamma={gamma:.4f}`.",
        "",
        "| split | method | error | identity error | improvement vs identity | ratio vs identity | alpha mean | alpha std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['split']} | {r['method']} | "
            f"{fmt(r['error'])} | {fmt(r['identity_error'])} | "
            f"{fmt(r['improvement_vs_identity'])} | {fmt(r['ratio_vs_identity'])} | "
            f"{fmt(r['alpha_mean'])} | {fmt(r['alpha_std'])} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception as e:
        raise RuntimeError("scikit-learn with HistGradientBoostingRegressor is required.") from e

    train = load_npz(args.train)
    val = load_npz(args.val)
    test = load_npz(args.test)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    stats = torch_stats_to_numpy(ckpt["stats"])
    model = build_model(ckpt, args.model, args.device)

    print("Predicting residuals...")
    x_train = normalized_input(train, stats)
    x_val = normalized_input(val, stats)
    x_test = normalized_input(test, stats)

    d_train = predict_delta(model, x_train, stats, args.batch_size, args.device)
    d_val = predict_delta(model, x_val, stats, args.batch_size, args.device)
    d_test = predict_delta(model, x_test, stats, args.batch_size, args.device)

    print("Selecting global alpha on validation...")
    alpha_global, val_global_error = select_global_alpha(
        val["z_current"].astype(np.float32),
        val["z_future"].astype(np.float32),
        d_val,
    )

    print("Training conditional alpha model on train per-sample oracle alpha...")
    train_true_delta = train["z_future"].astype(np.float32) - train["z_current"].astype(np.float32)
    y_alpha_train = oracle_alpha(d_train, train_true_delta)

    xa_train = compact_alpha_features(train, d_train)
    xa_val = compact_alpha_features(val, d_val)
    xa_test = compact_alpha_features(test, d_test)

    alpha_model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=args.seed,
    )
    alpha_model.fit(xa_train, y_alpha_train)

    alpha_val_raw = np.clip(alpha_model.predict(xa_val).astype(np.float32), 0.0, 1.0)
    alpha_test_raw = np.clip(alpha_model.predict(xa_test).astype(np.float32), 0.0, 1.0)

    print("Selecting safe mixture gamma on validation...")
    gamma, val_mix_error = select_mixing_gamma(
        val["z_current"].astype(np.float32),
        val["z_future"].astype(np.float32),
        d_val,
        alpha_global,
        alpha_val_raw,
    )

    rows = []
    rows += summarize_split("val", val, d_val, alpha_global, alpha_val_raw, gamma)
    rows += summarize_split("test", test, d_test, alpha_global, alpha_test_raw, gamma)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.model}_conditional_shrinkage_seed{args.seed}.csv"
    md_path = args.out_dir / f"{args.model}_conditional_shrinkage_seed{args.seed}.md"

    write_csv(csv_path, rows)
    write_md(md_path, rows, alpha_global, gamma)

    print(md_path)
    print(md_path.read_text())


if __name__ == "__main__":
    main()
