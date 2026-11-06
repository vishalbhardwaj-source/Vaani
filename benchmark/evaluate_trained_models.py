"""
Vaani: Evaluation Harness for Newly Trained Models
Evaluates:
- Model 1 (Denoiser) + Model 2 (Hinglish ASR Whisper+LoRA)
Across all 100 benchmark clips:
- Clean (25)
- Telephony 8 kHz (25)
- Telephony + 10 dB SNR (25)
- Telephony + 5 dB SNR (25)
Calculates exact Word Error Rate (WER) using jiwer.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import scipy.signal as signal
import torch
import jiwer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.denoiser_model import TelephonyDenoiser
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def clean_text(t: str) -> str:
    import re
    t = str(t).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())

def evaluate():
    print("=" * 70)
    print("EVALUATING NEWLY TRAINED MODEL PIPELINE ON RTX 5070 GPU")
    print("=" * 70)

    # 1. Load Model 1 (Denoiser)
    denoiser = TelephonyDenoiser()
    denoiser_path = PROJECT_ROOT / "models" / "denoiser_8khz.pt"
    denoiser.load_state_dict(torch.load(str(denoiser_path), map_location=device))
    denoiser.to(device).eval()

    # 2. Load Model 2 (Hinglish ASR)
    asr_dir = PROJECT_ROOT / "models" / "whisper_hinglish_lora"
    processor = WhisperProcessor.from_pretrained(str(asr_dir))
    base_asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    asr_model = PeftModel.from_pretrained(base_asr, str(asr_dir))
    asr_model.to(device).eval()

    # Load utterances
    csv_path = PROJECT_ROOT / "benchmark" / "utterances.csv"
    df = pd.read_csv(csv_path)

    audio_dir = PROJECT_ROOT / "benchmark" / "audio"
    conditions = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]

    results = []

    for cond in conditions:
        refs = []
        hyps_raw = []
        hyps_with_denoiser = []

        for _, row in df.iterrows():
            utt_id = row["id"]
            ref = str(row["utterance"]).strip()
            audio_path = audio_dir / f"{utt_id}_{cond}.wav"

            if not audio_path.exists():
                continue

            data, sr = sf.read(str(audio_path))
            orig_sr = sr

            # 1. Without Denoiser (Direct ASR)
            if sr != 16000:
                data_16k = signal.resample_poly(data, 16000, sr)
            else:
                data_16k = data

            feats = processor(data_16k, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                tokens = asr_model.generate(feats)
                pred_raw = processor.batch_decode(tokens, skip_special_tokens=True)[0]

            # 2. With Model 1 Denoiser Preprocessing
            if orig_sr != 8000:
                data_8k = signal.resample_poly(data, 8000, orig_sr)
            else:
                data_8k = data

            t_8k = torch.tensor(data_8k, dtype=torch.float32, device=device)
            cleaned_8k = denoiser.enhance(t_8k).cpu().numpy()
            cleaned_16k = signal.resample_poly(cleaned_8k, 16000, 8000)

            feats_cl = processor(cleaned_16k, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                tokens_cl = asr_model.generate(feats_cl)
                pred_cleaned = processor.batch_decode(tokens_cl, skip_special_tokens=True)[0]

            refs.append(clean_text(ref))
            hyps_raw.append(clean_text(pred_raw))
            hyps_with_denoiser.append(clean_text(pred_cleaned))

        wer_raw = jiwer.wer(refs, hyps_raw) * 100.0
        wer_cleaned = jiwer.wer(refs, hyps_with_denoiser) * 100.0

        results.append({
            "Acoustic Condition": cond,
            "Clips (N)": len(refs),
            "Direct ASR WER (%)": round(wer_raw, 2),
            "Denoiser + ASR WER (%)": round(wer_cleaned, 2),
            "WER Improvement": f"{-round(wer_raw - wer_cleaned, 2):+0.2f}%"
        })

    report_df = pd.DataFrame(results)
    print("\n" + report_df.to_string(index=False))
    print("=" * 70)
    return report_df

if __name__ == "__main__":
    evaluate()
