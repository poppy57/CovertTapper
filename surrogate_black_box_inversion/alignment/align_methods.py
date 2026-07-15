"""
Alignment method implementations:ModeltoModel.

Experiment 1.2 core module.Supports 8 (A0-A7),from tolayer MLP.

All methods inherit from BaseAligner and share a unified interface:
  - fit(h_surr, h_target): Calibration dataTraining/
  - transform(h_surr): transform surrogate hidden states into the target space
  - num_params: Trainingparameter
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Dict, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AlignResult:
    """Training"""
    method_name: str
    num_params: int
    train_mse: float
    val_mse: float
    train_epochs: int
    converged: bool


class BaseAligner(ABC, nn.Module):
    """Base class for alignment methods"""

    def __init__(self, src_dim: int, tgt_dim: int):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim

    @property
    def method_name(self) -> str:
        return self.__class__.__name__

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def fit(
        self,
        h_surr: torch.Tensor,
        h_target: torch.Tensor,
        val_surr: Optional[torch.Tensor] = None,
        val_target: Optional[torch.Tensor] = None,
    ) -> AlignResult:
        """Calibration dataalignment transform"""
        pass

    def transform(self, h_surr: torch.Tensor) -> torch.Tensor:
        """Alignment transform (forward pass)"""
        params = list(self.parameters())
        if params:
            h_surr = h_surr.to(params[0].dtype)
        return self.forward(h_surr)

    def summary(self) -> str:
        return f"{self.method_name}: src_dim={self.src_dim}, tgt_dim={self.tgt_dim}, params={self.num_params:,}"


class IdentityAligner(BaseAligner):
    """A0: No alignment; use surrogate hidden states directly"""

    def __init__(self, src_dim: int, tgt_dim: int):
        super().__init__(src_dim, tgt_dim)
        assert src_dim == tgt_dim, "Identity aligner requires same dimensions"

    def forward(self, h_surr: torch.Tensor) -> torch.Tensor:
        return h_surr

    def fit(self, h_surr, h_target, val_surr=None, val_target=None):
        with torch.no_grad():
            train_mse = float(((h_surr - h_target) ** 2).mean())
            val_mse = float(((val_surr - val_target) ** 2).mean()) if val_surr is not None else train_mse
        return AlignResult(self.method_name, 0, train_mse, val_mse, 0, True)


class MeanVarAligner(BaseAligner):
    """A1: mean-variance normalization"""

    def __init__(self, src_dim: int, tgt_dim: int):
        super().__init__(src_dim, tgt_dim)
        assert src_dim == tgt_dim, "MeanVar aligner requires same dimensions"
        self.register_buffer("src_mean", torch.zeros(src_dim))
        self.register_buffer("src_std", torch.ones(src_dim))
        self.register_buffer("tgt_mean", torch.zeros(tgt_dim))
        self.register_buffer("tgt_std", torch.ones(tgt_dim))

    def forward(self, h_surr: torch.Tensor) -> torch.Tensor:
        normalized = (h_surr - self.src_mean) / self.src_std.clamp(min=1e-8)
        return normalized * self.tgt_std + self.tgt_mean

    def fit(self, h_surr, h_target, val_surr=None, val_target=None):
        self.to(h_surr.device)
        with torch.no_grad():
            self.src_mean.copy_(h_surr.mean(dim=0))
            self.src_std.copy_(h_surr.std(dim=0))
            self.tgt_mean.copy_(h_target.mean(dim=0))
            self.tgt_std.copy_(h_target.std(dim=0))

            aligned = self.forward(h_surr)
            train_mse = float(((aligned - h_target) ** 2).mean())
            val_mse = train_mse
            if val_surr is not None:
                val_aligned = self.forward(val_surr)
                val_mse = float(((val_aligned - val_target) ** 2).mean())

        return AlignResult(self.method_name, 0, train_mse, val_mse, 0, True)


class OrthogonalProcrustes(BaseAligner):
    """A2: orthogonal Procrustes transform (SVD closed-form solution)"""

    def __init__(self, src_dim: int, tgt_dim: int):
        super().__init__(src_dim, tgt_dim)
        assert src_dim == tgt_dim, "Procrustes requires same dimensions"
        self.register_buffer("W", torch.eye(src_dim))

    def forward(self, h_surr: torch.Tensor) -> torch.Tensor:
        return h_surr @ self.W.T

    def fit(self, h_surr, h_target, val_surr=None, val_target=None):
        self.to(h_surr.device)
        with torch.no_grad():
            H = h_target.T @ h_surr  # (d, d)
            U, _, Vt = torch.linalg.svd(H.float())
            self.W.copy_((U @ Vt).to(self.W.dtype))

            aligned = self.forward(h_surr)
            train_mse = float(((aligned - h_target) ** 2).mean())
            val_mse = train_mse
            if val_surr is not None:
                val_aligned = self.forward(val_surr)
                val_mse = float(((val_aligned - val_target) ** 2).mean())

        return AlignResult(self.method_name, 0, train_mse, val_mse, 0, True)


class DiagonalAffine(BaseAligner):
    """A3: diagonal affine transform h_aligned = w * h + b"""

    def __init__(self, src_dim: int, tgt_dim: int):
        super().__init__(src_dim, tgt_dim)
        assert src_dim == tgt_dim, "Diagonal affine requires same dimensions"
        self.weight = nn.Parameter(torch.ones(src_dim))
        self.bias = nn.Parameter(torch.zeros(src_dim))

    def forward(self, h_surr: torch.Tensor) -> torch.Tensor:
        return h_surr * self.weight + self.bias

    def fit(self, h_surr, h_target, val_surr=None, val_target=None,
            lr=1e-3, epochs=500, patience=30):
        return _gradient_fit(self, h_surr, h_target, val_surr, val_target,
                             lr=lr, epochs=epochs, patience=patience)


class LowRankAffine(BaseAligner):
    """A4/A5: low-rank affine transform h_aligned = U V h + b"""

    def __init__(self, src_dim: int, tgt_dim: int, rank: int = 64):
        super().__init__(src_dim, tgt_dim)
        self.rank = rank
        self.U = nn.Parameter(torch.randn(tgt_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, src_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(tgt_dim))
        self._init_near_identity()

    def _init_near_identity(self):
        """initialize close to the identity mapping(dimension)"""
        r = min(self.rank, self.src_dim, self.tgt_dim)
        with torch.no_grad():
            self.U.zero_()
            self.V.zero_()
            for i in range(r):
                if i < self.tgt_dim and i < self.rank:
                    self.U[i, i] = 1.0
                if i < self.rank and i < self.src_dim:
                    self.V[i, i] = 1.0

    def forward(self, h_surr: torch.Tensor) -> torch.Tensor:
        return h_surr @ self.V.T @ self.U.T + self.bias

    def fit(self, h_surr, h_target, val_surr=None, val_target=None,
            lr=5e-4, epochs=800, patience=50):
        return _gradient_fit(self, h_surr, h_target, val_surr, val_target,
                             lr=lr, epochs=epochs, patience=patience)


class FullAffine(BaseAligner):
    """A6: full-matrix affine transform h_aligned = W h + b"""

    def __init__(self, src_dim: int, tgt_dim: int):
        super().__init__(src_dim, tgt_dim)
        self.linear = nn.Linear(src_dim, tgt_dim, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h_surr: torch.Tensor) -> torch.Tensor:
        return self.linear(h_surr)

    def fit(self, h_surr, h_target, val_surr=None, val_target=None,
            lr=1e-4, epochs=1000, patience=50, weight_decay=1e-4):
        return _gradient_fit(self, h_surr, h_target, val_surr, val_target,
                             lr=lr, epochs=epochs, patience=patience,
                             weight_decay=weight_decay)


class LRRAAligner(BaseAligner):
    """
    Low-Rank Residual Alignment (LRRA).

    g(h) = h + V_k (W_k V_k^T h + b_k)

    V_k: top-k right singular vectors of the residual matrix (frozen after SVD).
    W_k, b_k: trainable parameters in the k-dimensional subspace.
    Total trainable params: k^2 + k.
    """

    def __init__(self, src_dim: int, tgt_dim: int, rank: int = 64):
        super().__init__(src_dim, tgt_dim)
        assert src_dim == tgt_dim, "LRRA requires same dimensions"
        self.rank = rank
        self.register_buffer("V_k", torch.zeros(src_dim, rank))
        self.W_k = nn.Parameter(torch.zeros(rank, rank))
        self.b_k = nn.Parameter(torch.zeros(rank))

    def forward(self, h_surr: torch.Tensor) -> torch.Tensor:
        z = h_surr @ self.V_k                # project: (..., k)
        c = z @ self.W_k.T + self.b_k        # affine in subspace: (..., k)
        return h_surr + c @ self.V_k.T       # back-project + residual: (..., d)

    def fit(self, h_surr, h_target, val_surr=None, val_target=None,
            lr=1e-3, epochs=1000, patience=50):
        device = h_surr.device
        self.to(device)

        with torch.no_grad():
            residuals = (h_target - h_surr).float()
            U, S, Vt = torch.linalg.svd(residuals, full_matrices=False)
            k = min(self.rank, Vt.shape[0])
            self.V_k.zero_()
            self.V_k[:, :k] = Vt[:k].T.to(self.V_k.dtype)

            total_var = (S ** 2).sum()
            explained = (S[:k] ** 2).sum() / total_var
            logger.info(f"  [LRRA] SVD: rank={k}, top-{k} explains {explained*100:.1f}% of residual variance")

        return _gradient_fit(self, h_surr, h_target, val_surr, val_target,
                             lr=lr, epochs=epochs, patience=patience)


class MLPAligner(BaseAligner):
    """A7: layer MLP """

    def __init__(self, src_dim: int, tgt_dim: int, mid_dim: int = 1024, dropout: float = 0.1):
        super().__init__(src_dim, tgt_dim)
        self.net = nn.Sequential(
            nn.Linear(src_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, tgt_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h_surr: torch.Tensor) -> torch.Tensor:
        return self.net(h_surr)

    def fit(self, h_surr, h_target, val_surr=None, val_target=None,
            lr=5e-4, epochs=1000, patience=50, weight_decay=1e-5):
        return _gradient_fit(self, h_surr, h_target, val_surr, val_target,
                             lr=lr, epochs=epochs, patience=patience,
                             weight_decay=weight_decay)


def _gradient_fit(
    aligner: BaseAligner,
    h_surr: torch.Tensor,
    h_target: torch.Tensor,
    val_surr: Optional[torch.Tensor],
    val_target: Optional[torch.Tensor],
    lr: float = 1e-3,
    epochs: int = 500,
    patience: int = 30,
    weight_decay: float = 0.0,
) -> AlignResult:
    """TrainingWorkflow"""
    device = h_surr.device
    aligner = aligner.to(device)
    aligner.train()

    optimizer = torch.optim.AdamW(
        aligner.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        aligned = aligner(h_surr)
        loss = criterion(aligned, h_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(aligner.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_mse = loss.item()

        if val_surr is not None:
            aligner.eval()
            with torch.no_grad():
                val_aligned = aligner(val_surr)
                val_mse = criterion(val_aligned, val_target).item()
            aligner.train()
        else:
            val_mse = train_mse

        if val_mse < best_val_mse - 1e-8:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in aligner.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 100 == 0:
            logger.info(
                f"  [{aligner.method_name}] Epoch {epoch+1}/{epochs}: "
                f"train_mse={train_mse:.6f}, val_mse={val_mse:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if no_improve >= patience:
            logger.info(f"  [{aligner.method_name}] Early stop at epoch {epoch+1}")
            break

    converged = no_improve < patience
    if best_state is not None:
        aligner.load_state_dict(best_state)
    aligner.eval()

    with torch.no_grad():
        final_train = criterion(aligner(h_surr), h_target).item()
        final_val = best_val_mse if val_surr is not None else final_train

    return AlignResult(
        method_name=aligner.method_name,
        num_params=aligner.num_params,
        train_mse=final_train,
        val_mse=final_val,
        train_epochs=epoch + 1,
        converged=converged,
    )


# -- Factory function --

ALIGNER_REGISTRY: Dict[str, type] = {
    "A0_identity": IdentityAligner,
    "A1_meanvar": MeanVarAligner,
    "A2_procrustes": OrthogonalProcrustes,
    "A3_diagonal": DiagonalAffine,
    "A4_lowrank64": LowRankAffine,
    "A5_lowrank256": LowRankAffine,
    "A6_full": FullAffine,
    "A7_mlp": MLPAligner,
    "LRRA": LRRAAligner,
}


def create_aligner(method_id: str, src_dim: int, tgt_dim: int, **kwargs) -> BaseAligner:
    """
    Create an aligner instance.

    Args:
        method_id: A0_identity, A1_meanvar, ..., A7_mlp
        src_dim: Modeldimension
        tgt_dim: Modeldimension
    """
    if method_id == "A4_lowrank64":
        return LowRankAffine(src_dim, tgt_dim, rank=64)
    elif method_id == "A5_lowrank256":
        return LowRankAffine(src_dim, tgt_dim, rank=256)
    elif method_id == "A7_mlp":
        mid_dim = kwargs.get("mid_dim", 1024)
        dropout = kwargs.get("dropout", 0.1)
        return MLPAligner(src_dim, tgt_dim, mid_dim=mid_dim, dropout=dropout)
    elif method_id == "LRRA":
        rank = kwargs.get("rank", 64)
        return LRRAAligner(src_dim, tgt_dim, rank=rank)
    elif method_id in ALIGNER_REGISTRY:
        return ALIGNER_REGISTRY[method_id](src_dim, tgt_dim)
    else:
        raise ValueError(f"Unknown method: {method_id}. Available: {list(ALIGNER_REGISTRY.keys())}")


def get_all_method_ids() -> list:
    return list(ALIGNER_REGISTRY.keys())
