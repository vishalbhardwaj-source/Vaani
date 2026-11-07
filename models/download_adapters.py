"""
Vaani: LoRA Adapter Verification & Download Utility
Verifies the presence and cryptographic SHA256 integrity of fine-tuned Whisper-LoRA adapters
and denoiser weights in models/.
Enforces HTTPS-only downloads from pinned repositories and verifies hashes before use.
"""

import sys
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger("vaani.download_adapters")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "models" / "whisper_hinglish_lora"
DENOISER_FILE = BASE_DIR / "models" / "denoiser_8khz.pt"

# Cryptographically pinned SHA256 hashes for trained artifacts
PINNED_SHA256 = {
    "adapter_config.json": "59386082f6a5f8e95b593ae69066f77433aaa4d141d14c71cc767e53ad5b18cf",
    "adapter_model.safetensors": "f0e4fff4d6a9263e07730585368695886f3f3aa6dfbfcd6ea361e447276b42ca",
    "processor_config.json": "0ddb6d4341f4f0f0373c0243b9a8a60f914e90ea2794f5833e4a2c389f54e69f",
    "tokenizer.json": "f1ed71a6a9e8f1f7feecad691817e3be0ebd4e3bf515a372bde419e0e4cfb200",
    "tokenizer_config.json": "5a50fffcbaa5870b9cc47ee59336d499e0214c9a6bd8b890f2eec9f42ac4a3ef",
    "denoiser_8khz.pt": "96aa2dc0bc395c213e6795948eebed731394b65e7f360e48f42fff6a871418a5"
}

REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json"
]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_adapters(enforce_hash: bool = True) -> bool:
    """Check if all required adapter files exist and match pinned SHA256 hashes."""
    if not ADAPTER_DIR.exists():
        logger.error(f"[SECURITY] Adapter directory does not exist: {ADAPTER_DIR}")
        return False

    missing = []
    hash_mismatches = []
    for f in REQUIRED_FILES:
        fp = ADAPTER_DIR / f
        if not fp.exists() or fp.stat().st_size == 0:
            missing.append(f)
            continue
        if enforce_hash and f in PINNED_SHA256:
            actual_hash = compute_sha256(fp)
            if actual_hash != PINNED_SHA256[f]:
                hash_mismatches.append((f, PINNED_SHA256[f], actual_hash))

    if missing:
        logger.error(f"[INTEGRITY ERROR] Missing or empty required adapter files: {missing}")
        return False

    if hash_mismatches:
        for fname, exp, act in hash_mismatches:
            logger.error(f"[SECURITY HASH MISMATCH] {fname}: expected {exp}, got {act}")
        return False

    logger.info(f"All {len(REQUIRED_FILES)} Whisper-LoRA adapter files verified with SHA256 integrity.")
    return True


def verify_denoiser(enforce_hash: bool = True) -> bool:
    """Check if denoiser_8khz.pt exists and matches pinned SHA256 hash."""
    if not DENOISER_FILE.exists():
        logger.warning(f"Denoiser model file does not exist: {DENOISER_FILE}")
        return False
    if enforce_hash and "denoiser_8khz.pt" in PINNED_SHA256:
        actual = compute_sha256(DENOISER_FILE)
        if actual != PINNED_SHA256["denoiser_8khz.pt"]:
            logger.error(f"[SECURITY HASH MISMATCH] denoiser_8khz.pt: expected {PINNED_SHA256['denoiser_8khz.pt']}, got {actual}")
            return False
    logger.info("Denoiser weights verified with SHA256 integrity.")
    return True


def download_or_verify(repo_id: str = "vishalbhardwaj/vaani-whisper-hinglish-lora") -> bool:
    """
    Verifies local adapters with pinned SHA256, and if missing, attempts HTTPS download from HF Hub.
    Fails explicitly with exit code 1 on hash mismatch or network failure.
    """
    if verify_adapters(enforce_hash=True):
        print("[Vaani] Fine-tuned Whisper-LoRA weights are present and verified with SHA256 integrity.")
        return True

    print(f"[Vaani] Verified adapters missing or hash mismatched in {ADAPTER_DIR}.")
    print(f"[Vaani] Attempting secure HTTPS download from Hugging Face Hub ({repo_id})...")
    try:
        from huggingface_hub import snapshot_download
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(ADAPTER_DIR),
            local_dir_use_symlinks=False
        )
        if verify_adapters(enforce_hash=True):
            print("[Vaani] Successfully downloaded and cryptographically verified adapters.")
            return True
        else:
            logger.error("[SECURITY] Downloaded files failed cryptographic SHA256 verification!")
            return False
    except Exception as e:
        logger.error(f"Failed to securely download adapters from {repo_id}: {e}")
        print("[Vaani Error] Model adapters could not be verified or downloaded.")
        print("[Vaani Info] To generate fresh weights locally, run: python models/train_whisper_lora.py")
        return False


if __name__ == "__main__":
    success = download_or_verify()
    sys.exit(0 if success else 1)
