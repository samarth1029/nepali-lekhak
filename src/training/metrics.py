from __future__ import annotations
from typing import Iterable, Optional
import torch


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(
    data_loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: Optional[int] = None,
) -> float:
    total_loss = 0.0
    n = len(data_loader)
    if n == 0:
        return float("nan")
    limit = n if num_batches is None else min(num_batches, n)
    it = iter(data_loader)
    for i in range(limit):
        input_batch, target_batch = next(it)
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += float(loss.item())
    return total_loss / float(limit)


def evaluate_model(
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss