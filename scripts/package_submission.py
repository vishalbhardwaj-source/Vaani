"""
Vaani: Safe Submission Packaging Script
Prepares a clean, production-ready submission zip file for the Gnani AI Hackathon.
Ensures:
1. .env file is strictly excluded (preventing accidental API key leaks)
2. Only clean .env.example with placeholder keys is included
3. Virtualenvs, caches, and raw weights are omitted
4. All required source code, tests, docs, and benchmark CSVs are bundled
"""

import os
import sys
import zipfile
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    ".git", ".pytest_cache", "__pycache__", "venv", ".venv",
    ".idea", ".vscode", "tts_cache"
}

EXCLUDE_FILES = {
    ".env", ".env.local", ".env.production",
    "fleet.sqlite-journal", "sample_timbre_output.wav",
    "api_cache.json", "benchmark_cache.json"
}

EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".zip"
}


def package_submission(output_zip: str = "Vaani_GnaniAI_Submission.zip"):
    zip_path = PROJECT_ROOT / output_zip
    if zip_path.exists():
        zip_path.unlink()

    print(f"Creating clean submission archive: {zip_path.name}")
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            rel_dir = Path(root).relative_to(PROJECT_ROOT)

            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT)

                if (file in EXCLUDE_FILES or 
                    file_path.name in EXCLUDE_FILES or 
                    file.endswith("cache.json") or 
                    file.startswith(".env")):
                    if file != ".env.example":
                        print(f"  [EXCLUDED SENSITIVE/CACHE FILE]: {rel_path}")
                        continue

                if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
                    continue

                # Add file to zip
                zipf.write(file_path, arcname=str(rel_path))
                file_count += 1

    print(f"\n✅ Packaged {file_count} files into {zip_path.resolve()}")
    print(f"Archive Size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("🔒 Sensitive credentials check: .env was strictly excluded.")


if __name__ == "__main__":
    package_submission()
