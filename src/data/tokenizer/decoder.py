from typing import Iterable
from .exceptions import DecodeError

class BPEDecoder:
    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, token_ids: Iterable[int]) -> str:
        out = []
        for tid in token_ids:
            if tid not in self.vocab:
                raise DecodeError(f"Unknown token id: {tid}")
            tok = self.vocab[tid]
            if tok == "\n":
                if out and not "".join(out).endswith(" "):
                    out.append(" ")
                out.append("\n")
            elif tok.startswith("Ġ"):
                out.append(" " + tok[1:])
            else:
                out.append(tok)
        return "".join(out)
