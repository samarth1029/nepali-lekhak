import argparse
import logging
from src.utils.config_loader import ConfigLoader
from src.data.loader.config import DataLoaderConfig
from src.data.loader.dataloader import DataLoaderManager
from src.data.tokenizer.tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DataLoaderManager CLI")
    parser.add_argument("--config", type=str, default="configs/data/dataloader.yaml", help="Path to dataloader config YAML")
    args = parser.parse_args()

    cfg = ConfigLoader.load_config(args.config)["dataloader"]
    dl_config = DataLoaderConfig(**cfg)

    tokenizer = BPETokenizer()
    tokenizer.load(dl_config.tokenizer_path)

    with open(dl_config.data_file, encoding="utf-8") as f:
        full_text = f.read()
    token_ids = tokenizer.encode(full_text, allowed_special={'<|endoftext|>'})

    split_idx = int(len(token_ids) * dl_config.train_ratio)
    train_token_ids = token_ids[:split_idx]
    val_token_ids = token_ids[split_idx:]

    train_text = tokenizer.decode(train_token_ids)
    val_text = tokenizer.decode(val_token_ids)

    manager = DataLoaderManager(dl_config, tokenizer)
    train_loader, val_loader = manager.create_loaders(train_text, val_text, context_length=dl_config.max_length)

    logger.info("DataLoaderManager CLI completed successfully.")

if __name__ == "__main__":
    main()