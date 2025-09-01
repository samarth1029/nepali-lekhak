from pydantic import BaseModel
from typing import List

class TokenizerConfig(BaseModel):
    vocab_size: int
    min_frequency: int
    special_tokens: List[str]
    save_dir: str
    vocab_file: str
    merges_file: str