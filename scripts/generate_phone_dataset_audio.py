"""
Vaani: Phone Dataset Acoustic Generator & Synthesizer
Generates all 50 calibrated telephonic audio files listed in data/phone_dataset/phone_manifest.csv.
Models the exact acoustic environments documented in CONSENT_NOTE.md:
- SPK_A: Roadside / Two-wheeler (wind buffet + passing engine rumble, 10 dB SNR)
- SPK_B: Enclosed Cabin (diesel engine idle resonance, 12 dB SNR)
- SPK_C: Kitchen / Mechanical Exhaust Fan (continuous airflow & fan hum, 10 dB SNR)
- SPK_D: Street / Market Background (traffic chatter + horn reflections, 8 dB SNR)
- SPK_E: Quiet Handset Baseline (pure 8 kHz narrow-band cellular codec, 25 dB SNR)
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import scipy.signal as signal
import soundfile as sf
from gtts import gTTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_CSV = PROJECT_ROOT / "data" / "phone_dataset" / "phone_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "phone_dataset"


def butter_bandpass(lowcut: float = 300.0, highcut: float = 3400.0, fs: float = 8000.0, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def apply_telephony_filter(data: np.ndarray, fs: float = 8000.0) -> np.ndarray:
    b, a = butter_bandpass(300.0, 3400.0, fs=fs, order=4)
    return signal.filtfilt(b, a, data)


def resample_to_8k(data: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == 8000:
        return data
    from math import gcd
    g = gcd(orig_sr, 8000)
    up = 8000 // g
    down = orig_sr // g
    return signal.resample_poly(data, up, down)


def generate_environment_noise(environment: str, duration_sec: float, sr: int = 8000) -> Tuple[np.ndarray, float]:
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    if "Roadside" in environment or "Two-wheeler" in environment:
        # Wind buffet + passing traffic rumble
        white = np.random.normal(0, 1, n_samples)
        b_wind, a_wind = signal.butter(2, 0.08)
        wind = signal.lfilter(b_wind, a_wind, white)
        engine = 0.4 * np.sin(2 * np.pi * 55.0 * t) + 0.2 * np.sin(2 * np.pi * 110.0 * t)
        noise = 0.7 * wind + 0.3 * engine
        snr = 10.0
    elif "Cabin" in environment:
        # Low frequency diesel idle rumble + vehicle vibration
        rumble = (
            0.5 * np.sin(2 * np.pi * 42.0 * t) +
            0.3 * np.sin(2 * np.pi * 84.0 * t + 0.4) +
            0.2 * np.sin(2 * np.pi * 126.0 * t + 1.2)
        )
        white = np.random.normal(0, 1, n_samples)
        b_low, a_low = signal.butter(1, 0.04)
        vibe = signal.lfilter(b_low, a_low, white)
        noise = 0.7 * rumble + 0.3 * vibe
        snr = 12.0
    elif "Kitchen" in environment or "Exhaust" in environment:
        # High-frequency continuous fan hiss + 120Hz electrical hum
        white = np.random.normal(0, 1, n_samples)
        b_fan, a_fan = signal.butter(2, [0.15, 0.65], btype='band')
        fan = signal.lfilter(b_fan, a_fan, white)
        hum = 0.15 * np.sin(2 * np.pi * 120.0 * t)
        noise = 0.85 * fan + 0.15 * hum
        snr = 10.0
    elif "Street" in environment or "Market" in environment:
        # Chatter + intermittent horn reflections
        white = np.random.normal(0, 1, n_samples)
        b_pink, a_pink = signal.butter(1, 0.1)
        chatter = signal.lfilter(b_pink, a_pink, white)
        horn = 0.12 * np.sin(2 * np.pi * 480.0 * t) * (np.sin(2 * np.pi * 0.4 * t) > 0.85)
        noise = 0.8 * chatter + 0.2 * horn
        snr = 8.0
    else:  # Quiet Handset Baseline
        # Clean cellular line with mild background noise
        white = np.random.normal(0, 0.05, n_samples)
        noise = white
        snr = 25.0

    return noise, snr


def mix_noise_at_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    p_speech = np.mean(speech ** 2)
    p_noise = np.mean(noise ** 2)

    if p_speech < 1e-9 or p_noise < 1e-9:
        return speech

    p_noise_target = p_speech / (10.0 ** (target_snr_db / 10.0))
    scale_factor = np.sqrt(p_noise_target / p_noise)
    mixed = speech + noise * scale_factor

    max_val = np.max(np.abs(mixed))
    if max_val > 0.98:
        mixed = mixed / max_val * 0.95
    return mixed


def generate_all_phone_audio():
    if not MANIFEST_CSV.exists():
        print(f"Error: Manifest not found at {MANIFEST_CSV}")
        return

    df = pd.read_csv(MANIFEST_CSV)
    print(f"Loaded manifest with {len(df)} clips across {df['speaker_id'].nunique()} speakers.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for idx, row in df.iterrows():
        clip_id = row["clip_id"]
        speaker_id = row["speaker_id"]
        env = row["environment"]
        target_filename = row["audio_file"]
        text = row["transcript_target"]
        out_path = OUTPUT_DIR / target_filename

        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[{idx+1}/{len(df)}] Already exists: {target_filename}")
            success_count += 1
            continue

        print(f"[{idx+1}/{len(df)}] Synthesizing {clip_id} ({speaker_id} | {env}) -> {target_filename}...")
        try:
            tts = gTTS(text=text, lang="hi", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_f:
                tmp_path = tmp_f.name
            tts.save(tmp_path)

            raw_audio, sr = sf.read(tmp_path)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            if len(raw_audio.shape) > 1:
                raw_audio = np.mean(raw_audio, axis=1)

            # Resample to 8 kHz and bandpass
            audio_8k = resample_to_8k(raw_audio, sr)
            audio_filtered = apply_telephony_filter(audio_8k, fs=8000.0)

            # Mix environment noise
            duration = len(audio_filtered) / 8000.0
            noise, snr_target = generate_environment_noise(env, duration, sr=8000)
            final_audio = mix_noise_at_snr(audio_filtered, noise, snr_target)

            # Save as 16-bit PCM WAV (8 kHz Mono)
            sf.write(str(out_path), final_audio, 8000, subtype='PCM_16')
            success_count += 1
        except Exception as e:
            print(f"Error generating {clip_id}: {e}")

    print(f"\nCompleted! Generated {success_count}/{len(df)} telephonic audio clips in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_phone_audio()
