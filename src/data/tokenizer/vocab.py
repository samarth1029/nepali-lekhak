from typing import Dict, Tuple
from pydantic import BaseModel, Field, field_validator

class VocabModel(BaseModel):
    """Vocabulary and merges metadata."""
    vocab: Dict[int, str] = Field(default_factory=dict)
    merges: Dict[Tuple[int, int], int] = Field(default_factory=dict)

    @field_validator("vocab", mode="before")
    @classmethod
    def ensure_int_keys(cls, v):
        return {int(k): str(val) for k, val in v.items()}
