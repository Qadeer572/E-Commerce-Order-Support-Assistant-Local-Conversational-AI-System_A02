"""One CPU embedder per process, with no implicit network access."""
import json
from backend.config import MODEL_DIR


class LocalEmbedder:
    dimension = 384

    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        self.revision = json.loads((MODEL_DIR / "manifest.json").read_text())["embedding"]["revision"]
        self.model = SentenceTransformer(str(MODEL_DIR / "embedding"), device="cpu", local_files_only=True)
        self.tokenizer = self.model.tokenizer

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False, convert_to_numpy=True).tolist()


def load_generation_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer", local_files_only=True)


class ConservativeTokenizer:
    """Byte upper bound keeps safe fallback prompts bounded when assets are absent."""
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode("utf-8", errors="ignore")

    def apply_chat_template(self, messages, **kwargs):
        return [0] * sum(len(self.encode(m["content"])) + 64 for m in messages)
