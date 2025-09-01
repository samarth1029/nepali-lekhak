import argparse
from pathlib import Path
from src.utils.config_loader import ConfigLoader
from src.data.tokenizer.tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model/tokenizer.yaml")
    parser.add_argument("--text", type=str, help="Path to training text file (for training)")
    parser.add_argument("--test_sentence", type=str, help="Sentence to encode/decode")
    args = parser.parse_args()

    cfg = ConfigLoader.load_config(args.config)["tokenizer"]

    tokenizer = BPETokenizer()

    if args.text:
        with open(args.text, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer.train(
            text=text,
            vocab_size=cfg["vocab_size"],
            allowed_special=cfg.get("special_tokens", [])
        )
        save_dir = Path(cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = save_dir / cfg["vocab_file"]
        tokenizer.save(vocab_path)
        print(f"Tokenizer trained and saved to {vocab_path}")
    else:
        vocab_path = Path(cfg["save_dir"]) / cfg["vocab_file"]
        tokenizer.load(vocab_path)
        print(f"Tokenizer loaded from {vocab_path}")

    if args.test_sentence:
        encoded = tokenizer.encode(args.test_sentence, allowed_special=set(cfg.get("special_tokens", [])))
        decoded = tokenizer.decode(encoded)
        print("Encoded:", encoded)
        print("Decoded:", decoded)

if __name__ == "__main__":
    main()