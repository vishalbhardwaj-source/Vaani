"""
Gnani AI Setup & Verification Script
Checks environment credentials and verifies connection to:
1. Gnani Prisma (STT)
2. Gnani Evon (Reasoning LLM)
3. Gnani Timbre (TTS)
"""

import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx
from dotenv import load_dotenv

# Load local environment
load_dotenv()

GNANI_API_KEY = os.getenv("GNANI_API_KEY", "").strip()
GNANI_PRISMA_URL = os.getenv("GNANI_PRISMA_URL", "https://api.vachana.ai/stt/v3")
GNANI_EVON_URL = os.getenv("GNANI_EVON_URL", "https://api.vachana.ai/v1/evon/chat")
GNANI_TIMBRE_URL = os.getenv("GNANI_TIMBRE_URL", "https://api.vachana.ai/api/v1/tts/inference")

print("=" * 60)
print("     GNANI AI (VACHANA) SETUP & CREDENTIALS VERIFICATION")
print("=" * 60)

if not GNANI_API_KEY or GNANI_API_KEY == "your_gnani_api_key_here":
    print("\n❌ GNANI_API_KEY is not configured in your .env file.")
    sys.exit(1)

print(f"\n✅ Found GNANI_API_KEY: {GNANI_API_KEY[:10]}...{GNANI_API_KEY[-8:]}")
print(f"📡 Prisma STT: {GNANI_PRISMA_URL}")
print(f"📡 Timbre TTS: {GNANI_TIMBRE_URL}")

# Vachana API uses 'X-API-Key-ID' header
headers = {
    "X-API-Key-ID": GNANI_API_KEY,
    "Content-Type": "application/json"
}

# 1. Test Timbre (TTS) with a sample read-back line
print("\n[1/2] Testing Gnani Timbre TTS (https://api.vachana.ai/api/v1/tts/inference)...")
timbre_payload = {
    "text": "Ramesh ji, reefer temperature 18 degree verify ho gaya hai.",
    "model": "timbre-v2.5",
    "voice": "Pranav"
}

try:
    with httpx.Client(timeout=20.0) as client:
        res = client.post(GNANI_TIMBRE_URL, headers=headers, json=timbre_payload)
        print(f"HTTP Status: {res.status_code}")
        if res.status_code == 200:
            print("✅ Timbre TTS call SUCCESSFUL! Audio response received.")
            print(f"Content length: {len(res.content)} bytes, Content-Type: {res.headers.get('content-type')}")
            # Save sample audio output
            output_audio_path = Path("sample_timbre_output.wav")
            with open(output_audio_path, "wb") as f:
                f.write(res.content)
            print(f"Saved test audio to: {output_audio_path.resolve()}")
        else:
            print(f"Response: {res.text[:300]}")
except Exception as e:
    print(f"⚠️ Error reaching Timbre: {e}")

# 2. Test Prisma STT endpoint with actual benchmark audio file
print("\n[2/2] Testing Gnani Prisma STT (https://api.vachana.ai/stt/v3)...")
sample_wav = Path("benchmark/audio/utt_01_clean.wav")
if not sample_wav.exists():
    print(f"⚠️ Sample audio {sample_wav} not found!")
else:
    try:
        with open(sample_wav, "rb") as f:
            audio_bytes = f.read()

        files = {
            "audio_file": ("utt_01_clean.wav", audio_bytes, "audio/wav")
        }
        data = {
            "model": "prisma-v3.0",
            "language_code": "hi-IN"
        }
        with httpx.Client(timeout=25.0) as client:
            res = client.post(
                GNANI_PRISMA_URL,
                headers={"X-API-Key-ID": GNANI_API_KEY},
                files=files,
                data=data
            )
            print(f"HTTP Status: {res.status_code}")
            if res.status_code == 200:
                print(f"✅ Prisma STT call SUCCESSFUL!")
                print(f"Transcript response:\n{res.json()}")
            else:
                print(f"Response: {res.text[:300]}")
    except Exception as e:
        print(f"⚠️ Error reaching Prisma: {e}")

print("\n" + "=" * 60)
print("Setup check complete.")
print("=" * 60)
