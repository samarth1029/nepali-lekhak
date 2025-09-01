from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupCosine:
    total_steps: int
    warmup_steps: int
    initial_lr: float
    peak_lr: float
    min_lr: float

    def lr_at(self, global_step: int) -> float:
        # guard for degenerate cases
        total = max(1, self.total_steps)
        warm = max(0, min(self.warmup_steps, total))
        step = max(0, min(global_step, total))

        lr_increment = (self.peak_lr - self.initial_lr) / total
        if step < warm:
            return self.initial_lr + step * lr_increment
        # cosine decay after warmup
        progress = (step - warm) / max(1, (total - warm))
        return self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
