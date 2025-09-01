from pydantic import BaseModel

class DataLoaderConfig(BaseModel):
    batch_size: int = 4
    max_length: int = 256
    stride: int = 128
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0
    train_ratio: float = 0.9
    data_file: str = "data/corpus.txt"
    tokenizer_path: str = "artifacts/tokenizer/vocab1k.json"