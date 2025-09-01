import argparse
import logging
from src.inference.generate import TextGenerationEngine
from src.models.llm.nep_gpt import NepaliGPT
from src.utils.config_loader import ConfigLoader
from src.models.llm.config import LLMConfig
from src.data.tokenizer.config import TokenizerConfig
from src.data.tokenizer.tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Text Generation CLI")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save generated output")
    args = parser.parse_args()

    tokenizer_raw_cfg = ConfigLoader.load_config("configs/model/tokenizer.yaml")["tokenizer"]
    tokenizer_cfg = TokenizerConfig(**tokenizer_raw_cfg)
    raw_cfg = ConfigLoader.load_config("configs/model/model_config.yaml")["llm_model"]
    cfg = LLMConfig(**raw_cfg)
    model = NepaliGPT(cfg)

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_cfg.save_dir + "/" + tokenizer_cfg.vocab_file)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    engine = TextGenerationEngine(model, tokenizer, context_size=cfg.context_length, eos_token_id=eos_token_id)
    output = engine.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(output)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info(f"Generated output written to {args.output_file}")

if __name__ == "__main__":
    main()