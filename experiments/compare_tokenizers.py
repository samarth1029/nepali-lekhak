import tiktoken
from src.data.tokenizer.tokenizer import BPETokenizer
from src.utils.config_loader import ConfigLoader
import time
import os

cfg = ConfigLoader.load_config("configs/model/tokenizer.yaml")["tokenizer"]
tokenizer = BPETokenizer()
tokenizer.load(cfg["save_dir"] + "/" + cfg["vocab_file"])

tik = tiktoken.get_encoding("gpt2")
corpus_path = "data/corpus.txt"
if not os.path.exists(corpus_path):
    raise FileNotFoundError(f"{corpus_path} not found.")
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus_text = f.read()

N = 10

start = time.time()
for _ in range(N):
    token_ids_gpt = tik.encode(corpus_text, allowed_special={"<|endoftext|>"})
end = time.time()
tik_time = (end - start) / N

start = time.time()
for _ in range(N):
    token_ids = tokenizer.encode(corpus_text, allowed_special={"<|endoftext|>"})
end = time.time()
our_time = (end - start) / N


speedup = tik_time / our_time if our_time > 0 else float('inf')

len_gpt = len(token_ids_gpt)
len_our = len(token_ids)

if len_gpt < len_our:
    efficiency = f"Tiktoken is more efficient: {len_gpt=} < {len_our=} tokens"
elif len_gpt > len_our:
    efficiency = f"Our tokenizer is more efficient: {len_our=} < {len_gpt=} tokens"
else:
    efficiency = "Both tokenizers are equally efficient in token count."

results = (
    f"Tiktoken encoding: {tik_time:.6f} sec per call\n"
    f"Our tokenizer encoding: {our_time:.6f} sec per call\n"
    f"len(token_ids_gpt)={len_gpt}\n"
    f"len(token_ids)={len_our}\n"
    f"Speedup (tiktoken/our_tokenizer): {speedup:.2f}x\n"
    f"{'Our tokenizer is faster!' if speedup < 1 else 'tiktoken is faster!' if speedup > 1 else 'Both are equally fast.'}\n"
    f"{efficiency}\n"
)

print(results)
with open("reports/compare_tokenizers_results.txt", "w", encoding="utf-8") as f:
    f.write(results)