import re
from typing import List, Set, Dict, Tuple, Optional
from .exceptions import EncodeError

SPACE_TOKEN = "Ġ"

class BPEEncoder:
    """
    Byte Pair Encoding (BPE) Encoder for tokenizing text using a given vocabulary and merge rules.
    """

    def __init__(
        self,
        vocab: Dict[int, str],
        inv_vocab: Dict[str, int],
        merges: Dict[Tuple[int, int], int]
    ) -> None:
        """
        Initialize the encoder with vocabularies and merge rules.
        """
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.merges = merges

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """
        Encode the input text into a list of token IDs.

        Args:
            text (str): The input text to encode.
            allowed_special (Optional[Set[str]]): Set of special tokens to recognize.

        Returns:
            List[int]: List of token IDs.
        """
        if not isinstance(text, str):
            raise EncodeError("Input text must be a string.")

        if allowed_special is None:
            allowed_special = set()

        token_ids: List[int] = []

        # Handle special tokens
        if allowed_special:
            specials_sorted = sorted(allowed_special, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(s) for s in specials_sorted) + ")"
            last_index = 0
            for m in re.finditer(pattern, text):
                prefix = text[last_index:m.start()]
                if prefix:
                    token_ids.extend(self.encode(prefix, allowed_special=None))
                special_token = m.group(0)
                if special_token in self.inv_vocab:
                    token_ids.append(self.inv_vocab[special_token])
                else:
                    raise EncodeError(f"Special token '{special_token}' not in vocab.")
                last_index = m.end()
            text = text[last_index:]

        # Split text into tokens
        tokens: List[str] = []
        for i, line in enumerate(text.split("\n")):
            if i > 0:
                tokens.append("\n")
            for j, word in enumerate(line.split()):
                prefix = SPACE_TOKEN if j > 0 or i > 0 else ""
                tokens.append(prefix + word)

        # Map tokens to IDs
        for t in tokens:
            if t in self.inv_vocab:
                token_ids.append(self.inv_vocab[t])
            else:
                token_ids.extend(self._tokenize_with_bpe(t))
        return token_ids

    def _tokenize_with_bpe(self, token: str) -> List[int]:
        """
        Tokenize a string using BPE merges.

        Args:
            token (str): The token to encode.

        Returns:
            List[int]: List of token IDs.
        """
        ids = [self.inv_vocab.get(ch) for ch in token]
        if any(i is None for i in ids):
            missing = [ch for ch, i in zip(token, ids) if i is None]
            raise EncodeError(f"Characters not found in vocab: {missing}")

        changed = True
        while changed and len(ids) > 1:
            changed = False
            new_ids, i = [], 0
            while i < len(ids) - 1:
                pair = (ids[i], ids[i+1])
                if pair in self.merges:
                    new_ids.append(self.merges[pair])
                    i += 2
                    changed = True
                else:
                    new_ids.append(ids[i])
                    i += 1
            if i < len(ids):
                new_ids.append(ids[i])
            ids = new_ids
        return ids