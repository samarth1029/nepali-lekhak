# src/training/trainer.py
from __future__ import annotations
import os
import logging
from typing import Iterable, Tuple, List, Optional, Callable, Dict, Any
import torch

from src.inference.generate import TextGenerationEngine
from .config import TrainingConfig
from .schedulers import WarmupCosine
from .metrics import evaluate_model, calc_loss_batch
from .checkpoint import save_checkpoint, prune_old_checkpoints
from src.visualization.vis import plot_series

logger = logging.getLogger(__name__)
Callback = Callable[[Dict[str, Any]], None]


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        generator_factory: Optional[Callable[[torch.nn.Module, Any], TextGenerationEngine]] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Args:
            config: TrainingConfig (pydantic) object
            generator_factory: optional callable to create TextGenerationEngine
                signature (model, tokenizer) -> TextGenerationEngine
                if None, Trainer will instantiate TextGenerationEngine directly.
            callbacks: optional list of callbacks invoked after each evaluation when a checkpoint is saved
        """
        self.cfg = config
        self.generator_factory = generator_factory or (lambda model, tokenizer: TextGenerationEngine(
            model, tokenizer, context_size=model.pos_emb.weight.shape[0]
        ))
        self.callbacks = callbacks or []

    def train(
        self,
        model: torch.nn.Module,
        train_loader: Iterable,
        val_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        tokenizer,
        *,
        scheduler: Optional[WarmupCosine] = None,
        global_step: int = -1,
        best_val_loss: float = float("inf"),
    ) -> Tuple[List[float], List[float], List[int], List[float]]:
        """
        Main training loop. Returns (train_losses, val_losses, track_tokens_seen, track_lrs).
        """
        ckpt_dir = self.cfg.checkpointing.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # tracking lists
        train_losses: List[float] = []
        val_losses: List[float] = []
        track_tokens_seen: List[int] = []
        track_lrs: List[float] = []

        tokens_seen = 0

        # scheduler
        peak_lr = optimizer.param_groups[0]["lr"]
        total_training_steps = max(1, len(train_loader) * self.cfg.n_epochs)
        sched = scheduler or WarmupCosine(
            total_steps=total_training_steps,
            warmup_steps=self.cfg.scheduler.warmup_steps,
            initial_lr=self.cfg.scheduler.initial_lr,
            peak_lr=peak_lr,
            min_lr=self.cfg.scheduler.min_lr,
        )

        for epoch in range(self.cfg.n_epochs):
            model.train()
            epoch_tokens_seen, epoch_steps = 0, 0

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                global_step += 1
                epoch_steps += 1

                loss, lr, processed_tokens = self._train_step(
                    input_batch,
                    target_batch,
                    model,
                    optimizer,
                    device,
                    sched,
                    global_step,
                )

                tokens_seen += int(processed_tokens)
                epoch_tokens_seen += int(processed_tokens)
                track_lrs.append(lr)

                # periodic evaluation + checkpointing
                if global_step % self.cfg.evaluation.eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, self.cfg.evaluation.eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    
                    logger.info(
                        "Ep %d (Iter %06d): Train %.3f | Val %.3f | LR %.2e",
                        epoch + 1,
                        global_step,
                        train_loss,
                        val_loss,
                        lr,
                    )

                    # best checkpointing
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # weights-only for deployment
                        best_model_path = os.path.join(ckpt_dir, self.cfg.checkpointing.best_model_filename)
                        torch.save(model.state_dict(), best_model_path)
                        # full-state for exact resume
                        best_ckpt_path = os.path.join(ckpt_dir, self.cfg.checkpointing.best_checkpoint_filename)
                        save_checkpoint(best_ckpt_path, model, optimizer, epoch, global_step, best_val_loss)

                        # call callbacks with a useful context
                        ctx = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_val_loss": best_val_loss,
                            "best_model_path": best_model_path,
                            "best_checkpoint_path": best_ckpt_path,
                        }
                        for cb in self.callbacks:
                            try:
                                cb(ctx)
                            except Exception:
                                logger.exception("Callback failed")

            # end-of-epoch tasks: save + prune + sample generation
            epoch_path = os.path.join(ckpt_dir, self.cfg.checkpointing.epoch_pattern.format(epoch + 1))
            save_checkpoint(epoch_path, model, optimizer, epoch, global_step, best_val_loss)
            prune_old_checkpoints(ckpt_dir, pattern="epoch_*.pth", keep_last_n=self.cfg.checkpointing.keep_last_n_epochs)

            if self.cfg.sampling.enabled and self.cfg.sampling.start_context:
                self._generate_sample(model, tokenizer)

        final_path = os.path.join(ckpt_dir, self.cfg.checkpointing.final_checkpoint_filename)
        save_checkpoint(final_path, model, optimizer, epoch, global_step, best_val_loss)

        self._save_plots(ckpt_dir, track_tokens_seen, train_losses, val_losses, track_lrs)
        return train_losses, val_losses, track_tokens_seen, track_lrs


    def _train_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        sched: WarmupCosine,
        global_step: int,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Perform forward/backward/step for a single batch. Returns (loss, lr, tokens_processed).
        """
        # compute LR and set optimizer param groups
        lr = sched.lr_at(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # compute loss
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()

        # clip if past warmup
        if global_step >= sched.warmup_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        tokens_processed = int(input_batch.numel())
        return loss, lr, tokens_processed

    def _generate_sample(self, model: torch.nn.Module, tokenizer, output_file="artifacts/generated/training.txt") -> None:
        """
        Generate a short sample using TextGenerationEngine
        """
        try:
            model.eval()
            generator = self.generator_factory(model, tokenizer)
            with torch.no_grad():
                sample = generator.generate(
                    max_new_tokens=self.cfg.sampling.max_new_tokens,
                    prompt=self.cfg.sampling.start_context,
                )
                logger.info("Sample generation (epoch sample): %s", sample)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(sample)
                logger.info(f"Generated output written to {output_file}")
        except Exception:
            logger.exception("Sample generation failed")
        finally:
            model.train()

    def _save_plots(
        self,
        ckpt_dir: str,
        track_tokens_seen: List[int],
        train_losses: List[float],
        val_losses: List[float],
        track_lrs: List[float],
    ) -> None:
        """
        Save simple visualizations (loss vs tokens, val vs tokens, lr schedule).
        """
        try:
            if len(track_tokens_seen) > 0 and len(train_losses) == len(track_tokens_seen):
                plot_series(track_tokens_seen, train_losses, "Train Loss vs Tokens", "Tokens Seen", "Train Loss",
                            os.path.join(ckpt_dir, "train_loss.png"))
            if len(track_tokens_seen) > 0 and len(val_losses) == len(track_tokens_seen):
                plot_series(track_tokens_seen, val_losses, "Val Loss vs Tokens", "Tokens Seen", "Val Loss",
                            os.path.join(ckpt_dir, "val_loss.png"))
            if len(track_lrs) > 0:
                plot_series(list(range(len(track_lrs))), track_lrs, "Learning Rate Schedule", "Step", "LR",
                            os.path.join(ckpt_dir, "lr_schedule.png"))
        except Exception:
            logger.exception("Saving plots failed")
