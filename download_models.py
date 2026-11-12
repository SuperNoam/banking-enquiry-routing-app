"""Download the pinned MiniLM feature encoder. The analysis agent uses your API."""
import os

from common import ROOT, read_json

os.environ.setdefault("HF_HOME", str(ROOT / "work/hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def download_encoder():
    from huggingface_hub import snapshot_download
    cfg = read_json(ROOT / "config.json")["encoder"]
    return snapshot_download(cfg["model"], revision=cfg["revision"], local_dir=ROOT / cfg["path"],
                             allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*", "README.md"])


if __name__ == "__main__":
    print(download_encoder())
