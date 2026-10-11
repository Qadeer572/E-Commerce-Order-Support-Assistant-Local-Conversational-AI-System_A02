"""One-time network preparation. Runtime loads only these local files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import MODEL_DIR


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, target, revision, patterns in [
        ("sentence-transformers/all-MiniLM-L6-v2", "embedding", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", ["*.json", "*.txt", "*.safetensors", "1_Pooling/*"]),
        ("Qwen/Qwen2.5-1.5B-Instruct", "tokenizer", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", ["tokenizer*", "vocab.json", "merges.txt", "config.json"]),
    ]:
        snapshot_download(name, revision=revision, local_dir=MODEL_DIR / target, allow_patterns=patterns)
        manifest[target] = {"model": name, "revision": revision}
    AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer", local_files_only=True)
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared pinned local embedding weights and generation tokenizer")


if __name__ == "__main__":
    main()
