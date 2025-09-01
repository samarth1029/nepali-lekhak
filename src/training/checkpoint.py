from __future__ import annotations
from pathlib import Path
import glob
import os
import torch
from typing import Any, Dict, Optional


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def prune_old_checkpoints(checkpoint_dir: str | Path, pattern: str = "epoch_*.pth", keep_last_n: Optional[int] = 3) -> None:
    if keep_last_n is None:
        return
    files = sorted(glob.glob(os.path.join(str(checkpoint_dir), pattern)))
    for f in files[:-keep_last_n]:
        try:
            os.remove(f)
        except OSError:
            pass