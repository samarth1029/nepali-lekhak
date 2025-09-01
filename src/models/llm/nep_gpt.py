import logging
import torch
import torch.nn as nn
from src.utils.config_loader import ConfigLoader
from src.models.llm.config import LLMConfig
from src.models.llm.blocks import TransformerBlock
from src.models.llm.blocks import LayerNorm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NepaliGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop = nn.Dropout(cfg.drop_rate)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = LayerNorm(cfg.emb_dim)
        self.head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
    @staticmethod
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    @staticmethod
    def model_size(model, input_dtype=torch.float32):
        """
        Returns the total size of the model
        """
        total_params = 0
        total_grads = 0
        for param in model.parameters():
            param_size = param.numel()
            total_params += param_size
            if param.requires_grad:
                total_grads += param_size

        total_buffers = sum(buf.numel() for buf in model.buffers())
        element_size = torch.tensor(0, dtype=input_dtype).element_size()
        total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
        total_memory_gb = total_memory_bytes / (1024**3)
        return total_memory_gb

    @staticmethod
    def print_model_summary():
        raw_cfg = ConfigLoader.load_config("configs/model/model_config.yaml")["llm_model"]
        cfg = LLMConfig(**raw_cfg)
        model = NepaliGPT(cfg)
        total, trainable = NepaliGPT.count_parameters(model)
        size_mb = NepaliGPT.model_size(model)
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable parameters: {trainable:,}")
        logger.info(f"Total size of the model: {size_mb:.2f} GB")

if __name__ == "__main__":
    NepaliGPT.print_model_summary()