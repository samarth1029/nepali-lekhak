import argparse
from pathlib import Path
from src.utils.config_loader import ConfigLoader
from src.data.tokenizer.tokenizer import BPETokenizer
from google.cloud import storage
import tempfile
import os

def download_from_gcs(gcs_path: str) -> str:
    """Download a GCS file to a local temporary file and return local path."""
    if not gcs_path.startswith("gs://"):
        return gcs_path  # already local

    client = storage.Client()
    path_parts = gcs_path[5:].split("/", 1)
    bucket_name, blob_name = path_parts[0], path_parts[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.download_to_filename(temp_file.name)
    return temp_file.name

def upload_to_gcs(local_path: str, gcs_path: str):
    """Upload a local file to a GCS path."""
    if not gcs_path.startswith("gs://"):
        raise ValueError("gcs_path must start with gs://")
    client = storage.Client()
    bucket_name, blob_name = gcs_path[5:].split("/", 1)
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to {gcs_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model/tokenizer.yaml")
    parser.add_argument("--text", type=str, help="Path to training text file (for training)")
    parser.add_argument("--test_sentence", type=str, help="Sentence to encode/decode")
    parser.add_argument("--output_gcs", type=str, help="GCS path to save tokenizer.json")
    args = parser.parse_args()

    # Download config and text if they are GCS paths
    config_path = download_from_gcs(args.config)
    text_path = download_from_gcs(args.text) if args.text else None

    cfg = ConfigLoader.load_config(config_path)["tokenizer"]

    tokenizer = BPETokenizer()

    if text_path:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer.train(
            text=text,
            vocab_size=cfg["vocab_size"],
            allowed_special=cfg.get("special_tokens", [])
        )

        # Save locally first
        save_dir = Path(tempfile.mkdtemp())
        vocab_path = save_dir / cfg["vocab_file"]
        tokenizer.save(vocab_path)
        print(f"Tokenizer trained and saved locally to {vocab_path}")

        # Upload to GCS if provided
        if args.output_gcs:
            upload_to_gcs(str(vocab_path), args.output_gcs)
    else:
        vocab_path = Path(cfg["save_dir"]) / cfg["vocab_file"]
        tokenizer.load(vocab_path)
        print(f"Tokenizer loaded from {vocab_path}")

    if args.test_sentence:
        encoded = tokenizer.encode(args.test_sentence, allowed_special=set(cfg.get("special_tokens", [])))
        decoded = tokenizer.decode(encoded)
        print("Encoded:", encoded)
        print("Decoded:", decoded)

    # Clean up temporary files
    if config_path.startswith(tempfile.gettempdir()):
        os.remove(config_path)
    if text_path and text_path.startswith(tempfile.gettempdir()):
        os.remove(text_path)

if __name__ == "__main__":
    main()