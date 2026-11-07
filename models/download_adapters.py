"""
Vaani: LoRA Adapter Verification & Download Utility
Verifies the presence and integrity of fine-tuned Whisper-LoRA adapters in models/whisper_hinglish_lora.
If missing, downloads weights from Hugging Face repository or creates from checkpoint.
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger("vaani.download_adapters")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "models" / "whisper_hinglish_lora"
REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json"
]


def verify_adapters() -> bool:
    """Check if all required adapter files exist and have non-zero size."""
    if not ADAPTER_DIR.exists():
        logger.warning(f"Adapter directory does not exist: {ADAPTER_DIR}")
        return False

    missing = []
    for f in REQUIRED_FILES:
        fp = ADAPTER_DIR / f
        if not fp.exists() or fp.stat().st_size == 0:
            missing.append(f)

    if missing:
        logger.warning(f"Missing or empty adapter files: {missing}")
        return False

    logger.info(f"All {len(REQUIRED_FILES)} Whisper-LoRA adapter files verified in {ADAPTER_DIR}.")
    return True


def download_or_verify(repo_id: str = "vishalbhardwaj/vaani-whisper-hinglish-lora") -> bool:
    """
    Verifies local adapters, and if missing, attempts to download from Hugging Face Hub.
    """
    if verify_adapters():
        print("[Vaani] Fine-tuned Whisper-LoRA weights are present and verified.")
        return True

    print(f"[Vaani] Adapters missing in {ADAPTER_DIR}. Attempting download from HF Hub ({repo_id})...")
    try:
        from huggingface_hub import snapshot_download
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(ADAPTER_DIR),
            local_dir_use_symlinks=False
        )
        if verify_adapters():
            print("[Vaani] Successfully downloaded and verified adapters.")
            return True
    except Exception as e:
        logger.error(f"Failed to download adapters from {repo_id}: {e}")
        print(f"[Vaani] Note: Local training pipeline (models/train_whisper_lora.py) can be run to train new adapters.")
        return False


if __name__ == "__main__":
    success = download_or_verify()
    sys.exit(0 if success else 1)
