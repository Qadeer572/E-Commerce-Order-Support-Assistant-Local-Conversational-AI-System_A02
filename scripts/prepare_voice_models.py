"""Download the local voice models once: Vosk (ASR) and Piper (TTS). Runtime then works offline.

Writes models/voice_manifest.json with file names, sizes, SHA-256 hashes and source URLs.
Idempotent: files that already match the expected size/hash are skipped.
"""
import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.config import ASR_MODEL_DIR, MODEL_DIR  # noqa: E402
from backend.config import ASR_FINAL_ENABLED, ASR_FINAL_MODEL_DIR  # noqa: E402

VOSK_NAME = "vosk-model-en-us-0.22-lgraph"
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_NAME}.zip"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
# Sizes come from HTTP HEAD checks; the .onnx SHA-256 is Hugging Face's X-Linked-ETag.
DOWNLOADS = [
    {"name": f"{VOSK_NAME}.zip", "url": VOSK_URL, "size": 130557655, "sha256": None,
     "dest": MODEL_DIR / f"{VOSK_NAME}.zip"},
    {"name": "en_US-lessac-medium.onnx", "url": PIPER_BASE + "en_US-lessac-medium.onnx", "size": 63201294,
     "sha256": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
     "dest": MODEL_DIR / "piper" / "en_US-lessac-medium.onnx"},
    {"name": "en_US-lessac-medium.onnx.json", "url": PIPER_BASE + "en_US-lessac-medium.onnx.json", "size": 4885,
     "sha256": None, "dest": MODEL_DIR / "piper" / "en_US-lessac-medium.onnx.json"},
]
MANIFEST = MODEL_DIR / "voice_manifest.json"
WHISPER_REPO = "Systran/faster-whisper-small.en"
WHISPER_REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"files": {}}


def verified(item: dict, recorded: dict) -> bool:
    dest = item["dest"]
    if not dest.exists() or dest.stat().st_size != item["size"]:
        return False
    expected = item["sha256"] or recorded.get(item["name"], {}).get("sha256")
    return expected is None or sha256_of(dest) == expected


def download(item: dict) -> None:
    item["dest"].parent.mkdir(parents=True, exist_ok=True)
    partial = item["dest"].with_suffix(item["dest"].suffix + ".part")
    print(f"Downloading {item['name']} ({item['size'] / 1e6:.1f} MB)")
    urllib.request.urlretrieve(item["url"], partial)
    if partial.stat().st_size != item["size"]:
        raise RuntimeError(f"{item['name']}: size {partial.stat().st_size} != expected {item['size']}")
    if item["sha256"] and sha256_of(partial) != item["sha256"]:
        raise RuntimeError(f"{item['name']}: SHA-256 mismatch")
    partial.replace(item["dest"])


def main() -> None:
    manifest = load_manifest()
    for item in DOWNLOADS:
        if verified(item, manifest["files"]):
            print(f"OK (already verified): {item['name']}")
        else:
            download(item)
        manifest["files"][item["name"]] = {
            "size": item["dest"].stat().st_size, "sha256": sha256_of(item["dest"]), "url": item["url"]}
    if not ASR_MODEL_DIR.exists():
        print(f"Extracting {VOSK_NAME}")
        with zipfile.ZipFile(MODEL_DIR / f"{VOSK_NAME}.zip") as archive:
            archive.extractall(MODEL_DIR)
    if ASR_FINAL_ENABLED:
        from huggingface_hub import snapshot_download
        snapshot_download(WHISPER_REPO, revision=WHISPER_REVISION,
            local_dir=ASR_FINAL_MODEL_DIR,
            allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"])
        manifest["final_asr"] = {"repo": WHISPER_REPO, "revision": WHISPER_REVISION,
            "files": {name: {"size": (ASR_FINAL_MODEL_DIR / name).stat().st_size,
                             "sha256": sha256_of(ASR_FINAL_MODEL_DIR / name)}
                      for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")}}
    manifest["prepared_on"] = date.today().isoformat()
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
