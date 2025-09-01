from collections import Counter, deque
from typing import List, Tuple, Optional, Iterable, Dict
import logging
from .exceptions import TrainingError

logger = logging.getLogger(__name__)

BASE_VOCAB_SIZE = 256
SPACE_TOKEN = "Ġ"

class BPETrainer:
    """
    Byte Pair Encoding (BPE) Trainer for building vocabulary and merge rules.
    """

    def __init__(self) -> None:
        self.vocab: Dict[int, str] = {}
        self.inv_vocab: Dict[str, int] = {}
        self.merges: Dict[Tuple[int, int], int] = {}

    def train(
        self,
        text: str,
        vocab_size: int,
        allowed_special: Optional[Iterable[str]] = None
    ) -> None:
        """
        Train the BPE tokenizer on the given text.

        Args:
            text (str): Input text to train on.
            vocab_size (int): Desired vocabulary size.
            allowed_special (Optional[Iterable[str]]): Special tokens to include.
        """
        if not isinstance(text, str):
            raise TrainingError("Input text must be a string.")
        if vocab_size <= 0:
            raise TrainingError("vocab_size must be > 0")

        logger.info("Starting BPE training with vocab_size=%d", vocab_size)

        processed = self._preprocess_text(text)
        self._init_vocab(processed, allowed_special)

        token_ids = [self.inv_vocab[c] for c in processed]

        for new_id in range(len(self.vocab), vocab_size):
            most_freq = self._find_freq_pair(token_ids)
            if most_freq is None:
                logger.info("No more pairs to merge at vocab size %d", len(self.vocab))
                break
            token_ids = self._replace_pair(token_ids, most_freq, new_id)
            self.merges[most_freq] = new_id

        self._finalize_vocab()
        logger.info("Training finished: vocab=%d merges=%d", len(self.vocab), len(self.merges))

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text by replacing spaces with a special token.
        """
        processed_chars = []
        for i, ch in enumerate(text):
            if ch == " " and i != 0:
                processed_chars.append(SPACE_TOKEN)
            if ch != " ":
                processed_chars.append(ch)
        return "".join(processed_chars)

    def _init_vocab(self, processed: str, allowed_special: Optional[Iterable[str]]) -> None:
        """
        Initialize vocabulary and inverse vocabulary.
        """
        unique_chars = [chr(i) for i in range(BASE_VOCAB_SIZE)]
        unique_chars.extend([c for c in sorted(set(processed)) if c not in unique_chars])
        if SPACE_TOKEN not in unique_chars:
            unique_chars.append(SPACE_TOKEN)

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inv_vocab = {char: i for i, char in self.vocab.items()}

        if allowed_special:
            for token in allowed_special:
                if token not in self.inv_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inv_vocab[token] = new_id

    def _finalize_vocab(self) -> None:
        """
        Add merged tokens to vocab and inv_vocab.
        """
        for (p0, p1), nid in list(self.merges.items()):
            merged = self.vocab[p0] + self.vocab[p1]
            self.vocab[nid] = merged
            self.inv_vocab[merged] = nid

    @staticmethod
    def _find_freq_pair(token_ids: List[int]) -> Optional[Tuple[int, int]]:
        """
        Find the most frequent adjacent pair in token_ids.
        """
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None
        return max(pairs.items(), key=lambda x: x[1])[0]

    @staticmethod
    def _replace_pair(token_ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        Replace all occurrences of a pair in token_ids with new_id.
        """
        dq, replaced = deque(token_ids), []
        while dq:
            cur = dq.popleft()
            if dq and (cur, dq[0]) == pair:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(cur)
        return replaced