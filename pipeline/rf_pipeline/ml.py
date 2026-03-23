"""Machine-learning extensions: window autoencoder and inverse regressor.

Both models are gated behind an optional PyTorch import so the rest of the
pipeline degrades gracefully when PyTorch is not installed.

Extending
---------
Swap ``WindowAutoencoder`` for a different architecture by subclassing
``torch.nn.Module``, keeping the same ``encode_windows`` interface so that
``train_autoencoder`` remains unchanged.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except Exception:
    StandardScaler = None  # type: ignore[assignment,misc]
    PCA = None  # type: ignore[assignment]

from .config import AEConfig, InverseConfig, InverseModelInfo
from .metrics import build_window_feature_matrix


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_torch_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    if not _TORCH_AVAILABLE:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def maybe_compile(model: "nn.Module", enabled: bool) -> "nn.Module":  # type: ignore[name-defined]
    """Optionally apply ``torch.compile`` (requires PyTorch ≥ 2.0)."""
    if not _TORCH_AVAILABLE or not enabled:
        return model
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model)  # type: ignore[call-arg]
        except Exception:
            return model
    return model


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch (if available) random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDatasetTorch(Dataset):  # type: ignore[misc]
    """Sliding-window dataset that pairs every window with itself (autoencoder)."""

    def __init__(
        self,
        sequences: dict[str, np.ndarray],
        window: int,
        stride: int,
        noise_std: float = 0.0,
    ):
        self.samples: list[np.ndarray] = []
        self.labels: list[int] = []
        self.noise_std = float(noise_std)
        for lbl, name in enumerate(sorted(sequences.keys())):
            x = sequences[name]
            for i in range(0, len(x) - window + 1, stride):
                seg = x[i : i + window].T  # (C, W)
                self.samples.append(seg.astype(np.float32))
                self.labels.append(lbl)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        x = self.samples[idx].copy()
        y = self.samples[idx].copy()
        if self.noise_std > 0:
            x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
        return x, y, self.labels[idx]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WindowAutoencoder(nn.Module):  # type: ignore[misc]
    """1-D window autoencoder: encoder → latent z → decoder → reconstructed window."""

    def __init__(
        self, in_channels: int, window: int, latent_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window = window
        flat = in_channels * window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat),
        )

    def forward(self, x: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
        z = self.encoder(x)
        y = self.decoder(z).view(x.shape)
        return y, z

    @torch.no_grad()  # type: ignore[misc]
    def encode_windows(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x)


class MLPRegressorTorch(nn.Module):  # type: ignore[misc]
    """Lightweight MLP for the inverse-characterization regression task."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_autoencoder(
    metrics_by_unit: dict[str, pd.DataFrame],
    cfg: AEConfig,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """Train the window autoencoder and return per-unit latent descriptors.

    Each descriptor is the concatenation of [mean, std, P10, P90] of latent z
    across all sliding windows for that unit, giving a compact representation
    of the unit's RF behaviour over frequency.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for the autoencoder path.")
    device = torch.device(get_torch_device())
    sequences = {name: build_window_feature_matrix(df) for name, df in metrics_by_unit.items()}
    first = next(iter(sequences.values()))
    dataset = WindowDatasetTorch(sequences, cfg.window_size, cfg.stride, cfg.noise_std)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = WindowAutoencoder(
        first.shape[1], cfg.window_size, cfg.latent_dim, cfg.hidden_dim
    ).to(device)
    model = maybe_compile(model, cfg.compile_model)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    model.train()
    for epoch in range(cfg.epochs):
        running, count = 0.0, 0
        for xb, yb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred, _ = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]
        history.append({"epoch": epoch + 1, "loss": running / max(1, count)})

    pd.DataFrame(history).to_csv(output_dir / "autoencoder_training_history.csv", index=False)
    torch.save(model.state_dict(), output_dir / "window_autoencoder.pt")

    model.eval()
    descriptors: dict[str, np.ndarray] = {}
    for name, seq in sequences.items():
        windows = [
            seq[i : i + cfg.window_size].T
            for i in range(0, len(seq) - cfg.window_size + 1, cfg.stride)
        ]
        x = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
        z = model.encode_windows(x).detach().cpu().numpy()
        descriptors[name] = np.concatenate([
            z.mean(axis=0),
            z.std(axis=0),
            np.percentile(z, 10, axis=0),
            np.percentile(z, 90, axis=0),
        ]).astype(np.float64)
    return descriptors


def train_inverse_model(
    x: np.ndarray,
    y: np.ndarray,
    cfg: InverseConfig,
    output_dir: Path,
) -> InverseModelInfo:
    """Train the inverse regressor: topology+AE features → VF parameters."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for inverse modeling.")
    if StandardScaler is None or PCA is None:
        raise RuntimeError("scikit-learn is required for inverse modeling.")

    device = torch.device(get_torch_device())
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_s = x_scaler.fit_transform(x)
    y_s = y_scaler.fit_transform(y)

    pca = PCA(n_components=min(cfg.target_latent_dim, y_s.shape[1], y_s.shape[0]))
    y_lat = pca.fit_transform(y_s).astype(np.float32)

    x_t = torch.tensor(x_s, dtype=torch.float32)
    y_t = torch.tensor(y_lat, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(x_t, y_t)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))

    model = MLPRegressorTorch(x_t.shape[1], y_t.shape[1], cfg.hidden_dim).to(device)
    model = maybe_compile(model, cfg.compile_model)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        running, count = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]
        history.append({"epoch": epoch + 1, "loss": running / max(1, count)})

    pd.DataFrame(history).to_csv(output_dir / "inverse_training_history.csv", index=False)
    torch.save(model.state_dict(), output_dir / "inverse_regressor.pt")

    np.savez(
        output_dir / "inverse_preprocessing.npz",
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_mean=y_scaler.mean_,
        y_scale=y_scaler.scale_,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
    )
    return {
        "x_scaler_mean": x_scaler.mean_.tolist(),
        "x_scaler_scale": x_scaler.scale_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "latent_dim": int(y_t.shape[1]),
        "n_train": int(len(x)),
    }
