import torch
import logging

logger = logging.getLogger(__name__)

class TextGenerationEngine:
    def __init__(self, model, tokenizer, context_size: int, eos_token_id: int = None):
        """
        Args:
            model: The trained language model.
            tokenizer: The tokenizer instance.
            context_size (int): Maximum context window size.
            eos_token_id (int, optional): End-of-sequence token ID.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.eos_token_id = eos_token_id

    def text_to_token_ids(self, text: str) -> torch.Tensor:
        """Convert text to token IDs tensor with batch dimension."""
        encoded = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        logger.debug(f"Encoded text '{text}' to token IDs: {encoded}")
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    def token_ids_to_text(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs tensor (with batch) back to text."""
        flat = token_ids.squeeze(0)
        text = self.tokenizer.decode(flat.tolist()).replace("Ġ", " ")
        logger.debug(f"Decoded token IDs {flat.tolist()} to text: '{text}'")
        return text

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_k: int = None,
    ) -> str:
        """
        Generate text from a prompt using the model.

        Args:
            prompt (str): Input text prompt.
            max_new_tokens (int): Number of tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int, optional): Top-k sampling.

        Returns:
            str: Generated text.
        """
        self.model.eval()
        idx = self.text_to_token_ids(prompt)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size :]
            with torch.no_grad():
                logits = self.model(idx_cond)
            print(logits.shape)
            logits = logits[:, -1, :]  # (batch, vocab_size)

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_val, torch.full_like(logits, float('-inf')), logits)

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if self.eos_token_id is not None and (idx_next == self.eos_token_id).all():
                logger.info("EOS token generated, stopping early.")
                break

            idx = torch.cat((idx, idx_next), dim=1)
        return self.token_ids_to_text(idx)