import argparse
import logging
import os
import torch

from src.training.trainer import Trainer
from src.training.config import TrainingConfig
from src.training.checkpoint import load_checkpoint
from src.models.llm.nep_gpt import NepaliGPT
from src.utils.config_loader import ConfigLoader
from src.models.llm.config import LLMConfig
from src.data.tokenizer.config import TokenizerConfig
from src.data.tokenizer.tokenizer import BPETokenizer
from src.data.loader.config import DataLoaderConfig
from src.data.loader.dataloader import DataLoaderManager


def main():
    parser = argparse.ArgumentParser(description="NepaliLekhak Training CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config_tokenizer", type=str, default="configs/model/tokenizer.yaml", help="Path to tokenizer config")
    parser.add_argument("--config_model", type=str, default="configs/model/model_config.yaml", help="Path to model config")
    parser.add_argument("--config_dataloader", type=str, default="configs/data/dataloader.yaml", help="Path to data loader config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting training with config: %s", args.config)

    # --- Tokenizer ---
    tokenizer_raw_cfg = ConfigLoader.load_config(args.config_tokenizer)["tokenizer"]
    tokenizer_cfg = TokenizerConfig(**tokenizer_raw_cfg)
    tokenizer = BPETokenizer()
    vocab_path = os.path.join(tokenizer_cfg.save_dir, tokenizer_cfg.vocab_file)
    tokenizer.load(vocab_path)

    # --- Training config ---
    cfg_dict = ConfigLoader.load_config(args.config)
    cfg = TrainingConfig(**cfg_dict)

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # --- Model ---
    raw_cfg = ConfigLoader.load_config(args.config_model)["llm_model"]
    cfg_llm = LLMConfig(**raw_cfg)
    model = NepaliGPT(cfg_llm).to(device)

    # --- Data ---
    cfg_dl = ConfigLoader.load_config(args.config_dataloader)["dataloader"]
    dl_config = DataLoaderConfig(**cfg_dl)

    with open(dl_config.data_file, encoding="utf-8") as f:
        full_text = f.read()

    split_idx = int(len(full_text) * dl_config.train_ratio)
    train_text, val_text = full_text[:split_idx], full_text[split_idx:]

    manager = DataLoaderManager(dl_config, tokenizer)
    train_loader, val_loader = manager.create_loaders(
        train_text, val_text, context_length=dl_config.max_length
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # --- Resume if requested ---
    global_step, best_val_loss = -1, float("inf")
    if args.resume and os.path.exists(args.resume):
        logger.info("Resuming from checkpoint: %s", args.resume)
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        global_step = checkpoint.get("global_step", -1)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # --- Train ---
    trainer = Trainer(cfg)
    trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        tokenizer=tokenizer,
        global_step=global_step,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    main()
