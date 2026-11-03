"""
Saarthi: Telephony & SNR Noise Degradation Pipeline
Simulates realistic telephonic audio degradation:
1. Resampling to 8 kHz (narrow-band telephony).
2. Applying 300–3400 Hz Butterworth bandpass filter.
3. Mixing calibrated truck cabin / engine noise at exact target SNRs (10 dB and 5 dB):
   SNR_dB = 10 * log10(P_speech / P_noise)
"""

import os
from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
from typing import Tuple

AUDIO_DIR = Path(__file__).resolve().parent / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def butter_bandpass(lowcut: float = 300.0, highcut: float = 3400.0, fs: float = 8000.0, order: int = 4):
    """Design a digital Butterworth bandpass filter for telephony passband."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data: np.ndarray, fs: float = 8000.0) -> np.ndarray:
    """Apply 300–3400 Hz telephony bandpass filter to 8 kHz audio."""
    b, a = butter_bandpass(300.0, 3400.0, fs=fs, order=4)
    # Use filtfilt for zero-phase distortion
    filtered = signal.filtfilt(b, a, data)
    return filtered


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int = 8000) -> np.ndarray:
    """Resample audio signal using scipy resample_poly."""
    if orig_sr == target_sr:
        return data
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return signal.resample_poly(data, up, down)


def generate_truck_noise(duration_sec: float, sr: int = 8000) -> np.ndarray:
    """
    Synthesize realistic Indian truck cabin noise:
    - Low-frequency diesel engine rumble (combustion stroke harmonics at ~45Hz, 90Hz, 135Hz)
    - Road vibration & tyre rumble (shaped pink noise)
    - Distant acoustic horn reflections
    """
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # 1. Engine diesel rumble harmonics
    rumble = (
        0.5 * np.sin(2 * np.pi * 42.0 * t) +
        0.3 * np.sin(2 * np.pi * 84.0 * t + 0.4) +
        0.2 * np.sin(2 * np.pi * 126.0 * t + 1.2) +
        0.1 * np.sin(2 * np.pi * 210.0 * t)
    )

    # 2. Road surface pink noise
    white = np.random.normal(0, 1, n_samples)
    b_pink, a_pink = signal.butter(1, 0.05)
    pink = signal.lfilter(b_pink, a_pink, white)
    pink /= (np.std(pink) + 1e-9)

    # 3. Occasional highway horn resonance in cabin
    horn = 0.08 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 0.3 * t) > 0.8)

    combined = 0.6 * rumble + 0.35 * pink + 0.05 * horn
    return combined


def mix_noise_at_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """
    Mix speech and noise at an exact target SNR (dB).
    Formula: SNR_dB = 10 * log10(P_speech / P_noise)
    => P_noise_target = P_speech / (10 ^ (target_snr_db / 10))
    => noise_scaled = noise * sqrt(P_noise_target / P_noise_actual)
    """
    p_speech = np.mean(speech ** 2)
    p_noise = np.mean(noise ** 2)

    if p_speech < 1e-9 or p_noise < 1e-9:
        return speech

    # Target noise power
    p_noise_target = p_speech / (10.0 ** (target_snr_db / 10.0))
    scale_factor = np.sqrt(p_noise_target / p_noise)
    
    scaled_noise = noise * scale_factor
    mixed = speech + scaled_noise

    # Normalize to avoid hard clipping (-1.0 to 1.0)
    max_val = np.max(np.abs(mixed))
    if max_val > 0.98:
        mixed = mixed / max_val * 0.95

    return mixed


def process_audio_file(
    input_wav_path: Path,
    output_prefix: str
) -> dict:
    """
    Takes a clean recording and produces 4 conditions:
    1. clean (16 kHz)
    2. telephony_8khz (8 kHz bandpassed)
    3. telephony_10db (8 kHz bandpassed + 10 dB SNR truck noise)
    4. telephony_5db (8 kHz bandpassed + 5 dB SNR loud cabin noise)
    """
    data, sr = sf.read(str(input_wav_path))
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # Convert stereo to mono

    out_paths = {}

    # 1. Clean
    clean_path = AUDIO_DIR / f"{output_prefix}_clean.wav"
    sf.write(str(clean_path), data, sr)
    out_paths["clean"] = clean_path

    # Resample to 8 kHz & apply 300-3400 Hz bandpass
    data_8k = resample_audio(data, sr, target_sr=8000)
    data_telephony = apply_bandpass_filter(data_8k, fs=8000.0)

    # 2. Telephony 8 kHz clean
    tel_path = AUDIO_DIR / f"{output_prefix}_telephony_8khz.wav"
    sf.write(str(tel_path), data_telephony, 8000)
    out_paths["telephony_8khz"] = tel_path

    # Generate matching noise
    duration = len(data_telephony) / 8000.0
    noise_8k = generate_truck_noise(duration, sr=8000)

    # 3. Telephony + 10 dB SNR
    mixed_10db = mix_noise_at_snr(data_telephony, noise_8k, target_snr_db=10.0)
    path_10db = AUDIO_DIR / f"{output_prefix}_telephony_10db.wav"
    sf.write(str(path_10db), mixed_10db, 8000)
    out_paths["telephony_10db"] = path_10db

    # 4. Telephony + 5 dB SNR (Severe highway noise)
    mixed_5db = mix_noise_at_snr(data_telephony, noise_8k, target_snr_db=5.0)
    path_5db = AUDIO_DIR / f"{output_prefix}_telephony_5db.wav"
    sf.write(str(path_5db), mixed_5db, 8000)
    out_paths["telephony_5db"] = path_5db

    return out_paths


def synthesize_benchmark_audio(utterances_csv_path: Path):
    """Generate audio files for all benchmark utterances using gTTS and degrade them."""
    import pandas as pd
    from gtts import gTTS

    df = pd.read_csv(utterances_csv_path)
    print(f"Generating audio for {len(df)} utterances across 4 conditions (100 clips total)...")

    for idx, row in df.iterrows():
        utt_id = row["id"]
        text = row["utterance"]
        raw_clean_path = AUDIO_DIR / f"{utt_id}_raw.wav"
        
        # Synthesize clean speech if not present
        if not raw_clean_path.exists():
            mp3_temp = AUDIO_DIR / f"{utt_id}_temp.mp3"
            tts = gTTS(text=text, lang="hi", slow=False)
            tts.save(str(mp3_temp))
            # Convert mp3 to wav via soundfile/scipy or ffmpeg
            os.system(f'ffmpeg -y -i "{mp3_temp}" -ar 16000 -ac 1 "{raw_clean_path}" -loglevel quiet')
            if mp3_temp.exists():
                mp3_temp.unlink()

        # Process through telephony and SNR conditions
        process_audio_file(raw_clean_path, utt_id)

    print("Audio generation complete: 100 clips generated.")


if __name__ == "__main__":
    csv_file = Path(__file__).resolve().parent / "utterances.csv"
    synthesize_benchmark_audio(csv_file)
