from pathlib import Path
import json
import logging
from .trainer import BPETrainer
from .encoder import BPEEncoder
from .decoder import BPEDecoder
from .exceptions import VocabLoadError

logger = logging.getLogger(__name__)

class BPETokenizer:
    def __init__(self):
        self.vocab, self.inv_vocab, self.merges = {}, {}, {}
        self.encoder, self.decoder = None, None

    def train(self, text: str, vocab_size: int, allowed_special=None):
        trainer = BPETrainer()
        trainer.train(text, vocab_size, allowed_special)
        self.vocab, self.inv_vocab, self.merges = trainer.vocab, trainer.inv_vocab, trainer.merges
        self.encoder = BPEEncoder(self.vocab, self.inv_vocab, self.merges)
        self.decoder = BPEDecoder(self.vocab)

    def encode(self, text: str, allowed_special=None):
        if not self.encoder:
            raise RuntimeError("Tokenizer not trained or loaded")
        return self.encoder.encode(text, allowed_special)

    def decode(self, token_ids):
        if not self.decoder:
            raise RuntimeError("Tokenizer not trained or loaded")
        return self.decoder.decode(token_ids)

    def save(self, path: Path | str):
        data = {
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": [{"pair": list(p), "new_id": nid} for p, nid in self.merges.items()],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path | str):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.vocab = {int(k): v for k, v in data["vocab"].items()}
            self.inv_vocab = {v: int(k) for k, v in self.vocab.items()}
            self.merges = {tuple(m["pair"]): m["new_id"] for m in data["merges"]}
        except Exception as e:
            raise VocabLoadError(f"Failed to load vocab: {e}")
        self.encoder = BPEEncoder(self.vocab, self.inv_vocab, self.merges)
        self.decoder = BPEDecoder(self.vocab)
        logger.info("Loaded tokenizer with %d vocab entries", len(self.vocab))