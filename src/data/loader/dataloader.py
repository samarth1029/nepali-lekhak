from torch.utils.data import DataLoader
import logging
from src.data.loader.config import DataLoaderConfig
from src.data.loader.dataset import TextChunkDataset

logger = logging.getLogger(__name__)

class DataLoaderManager:
    def __init__(self, config: DataLoaderConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def split_data(self, text: str):
        total_tokens = len(self.tokenizer.encode(text, allowed_special={'<|endoftext|>'}))
        train_size = int(total_tokens * self.config.train_ratio)
        val_size = total_tokens - train_size
        return train_size, val_size, total_tokens

    def create_loaders(self, train_text: str, val_text: str, context_length: int):
        train_size, val_size, total_tokens = self.split_data(train_text + val_text)
        logger.info(f"Total tokens: {total_tokens}")

        if train_size < context_length:
            logger.warning("Not enough tokens for the training loader. "
                           "Try to lower the context_length or increase the training_ratio.")
        if val_size < context_length:
            logger.warning("Not enough tokens for the validation loader. "
                           "Try to lower the context_length or decrease the training_ratio.")

        train_dataset = TextChunkDataset(
            train_text, self.tokenizer, self.config.max_length, self.config.stride
        )
        val_dataset = TextChunkDataset(
            val_text, self.tokenizer, self.config.max_length, self.config.stride
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            drop_last=self.config.drop_last,
            num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers
        )

        train_tokens = sum(input_batch.numel() for input_batch, _ in train_loader)
        val_tokens = sum(input_batch.numel() for input_batch, _ in val_loader)

        logger.info(f"Training tokens: {train_tokens}")
        logger.info(f"Validation tokens: {val_tokens}")
        logger.info(f"All tokens: {train_tokens + val_tokens}")

        return train_loader, val_loader