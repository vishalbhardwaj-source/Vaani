"""
Vaani: Empirical Evaluation of Trained Acoustic Pipeline
Evaluates Model 1 (8 kHz Front-End Denoiser) and Model 2 (Hinglish Whisper-LoRA ASR).
Computes Word Error Rate (WER) across all 4 channel conditions, with rigorous
distinction between Training Set and Held-Out Validation/Test Sets.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import scipy.signal as signal
import torch
import jiwer
import re
from typing import Dict, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.denoiser_model import TelephonyDenoiser
from core.asr_model import get_asr_service

device = "cuda" if torch.cuda.is_available() else "cpu"

# 7 Held-out test utterances (representing ~28% held-out unseen evaluation set)
HELD_OUT_UTT_IDS = {"utt_03", "utt_07", "utt_11", "utt_15", "utt_19", "utt_22", "utt_25"}


def clean_text(t: str) -> str:
    import re
    t = str(t).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def check_number_survival(ref: str, hyp: str, row_dict: Dict[str, Any]) -> Tuple[int, int]:
    """
    Checks if numbers (temperature and distance) present in reference survive in hypothesis.
    Returns: (survived_count, total_expected_count)
    """
    from gnani.client import parse_hindi_number
    expected = []
    if pd.notna(row_dict.get("true_temperature_c")):
        try:
            expected.append(float(row_dict["true_temperature_c"]))
        except Exception:
            pass
    if pd.notna(row_dict.get("true_distance_km")):
        try:
            expected.append(float(row_dict["true_distance_km"]))
        except Exception:
            pass

    if not expected:
        return 0, 0

    hyp_lower = hyp.lower()
    survived = 0
    for num in expected:
        num_str = f"{num:g}"
        abs_str = f"{abs(num):g}"
        if re.search(rf"\b{re.escape(num_str)}\b", hyp_lower) or re.search(rf"\b{re.escape(abs_str)}\b", hyp_lower):
            survived += 1
        else:
            parsed = parse_hindi_number(hyp_lower)
            if parsed is not None and abs(parsed - num) < 0.1:
                survived += 1

    return survived, len(expected)


def evaluate():
    print("=" * 75)
    print("EVALUATING TRAINED MODEL PIPELINE ON HELD-OUT & IN-DOMAIN SETS")
    print("=" * 75)

    # 1. Load Model 1 (Denoiser)
    denoiser = TelephonyDenoiser()
    denoiser_path = PROJECT_ROOT / "models" / "denoiser_8khz.pt"
    if denoiser_path.exists():
        denoiser.load_state_dict(torch.load(str(denoiser_path), map_location=device))
    denoiser.to(device).eval()

    # 2. Load Model 2 (Hinglish ASR via unified service)
    asr_service = get_asr_service()
    processor = asr_service.processor
    asr_model = asr_service.model

    # Load utterances
    csv_path = PROJECT_ROOT / "benchmark" / "utterances.csv"
    df = pd.read_csv(csv_path)

    audio_dir = PROJECT_ROOT / "benchmark" / "audio"
    conditions = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]

    summary_rows = []

    for cond in conditions:
        refs_all, hyps_raw_all, hyps_denoised_all = [], [], []
        refs_heldout, hyps_raw_heldout, hyps_denoised_heldout = [], [], []
        refs_indomain, hyps_raw_indomain, hyps_denoised_indomain = [], [], []

        h_surv_raw, h_tot_raw = 0, 0
        h_surv_den, h_tot_den = 0, 0
        i_surv_raw, i_tot_raw = 0, 0
        i_surv_den, i_tot_den = 0, 0

        for _, row in df.iterrows():
            utt_id = row["id"]
            ref = str(row["utterance"]).strip()
            audio_path = audio_dir / f"{utt_id}_{cond}.wav"

            if not audio_path.exists():
                continue

            data, sr = sf.read(str(audio_path))

            # 1. Without Denoiser (Direct ASR)
            if sr != 16000:
                audio_16k_raw = signal.resample_poly(data, 16000, sr)
            else:
                audio_16k_raw = data

            in_raw = processor(audio_16k_raw, sampling_rate=16000, return_tensors="pt")
            in_raw = {k: v.to(device) for k, v in in_raw.items()}
            with torch.no_grad():
                gen_raw = asr_model.generate(**in_raw, max_new_tokens=128, language="en", task="transcribe")
                hyp_raw = processor.batch_decode(gen_raw, skip_special_tokens=True)[0].strip()

            # 2. With Model 1 Denoiser
            if sr != 8000:
                audio_8k = signal.resample_poly(data, 8000, sr)
            else:
                audio_8k = data

            tensor_8k = torch.tensor(audio_8k, dtype=torch.float32, device=device)
            cleaned_8k = denoiser.enhance(tensor_8k).cpu().numpy()
            cleaned_16k = signal.resample_poly(cleaned_8k, 16000, 8000)

            in_cleaned = processor(cleaned_16k, sampling_rate=16000, return_tensors="pt")
            in_cleaned = {k: v.to(device) for k, v in in_cleaned.items()}
            with torch.no_grad():
                gen_cleaned = asr_model.generate(**in_cleaned, max_new_tokens=128, language="en", task="transcribe")
                hyp_denoised = processor.batch_decode(gen_cleaned, skip_special_tokens=True)[0].strip()

            refs_all.append(clean_text(ref))
            hyps_raw_all.append(clean_text(hyp_raw))
            hyps_denoised_all.append(clean_text(hyp_denoised))

            row_dict = row.to_dict()
            s_raw, t_raw = check_number_survival(ref, hyp_raw, row_dict)
            s_den, t_den = check_number_survival(ref, hyp_denoised, row_dict)

            if utt_id in HELD_OUT_UTT_IDS:
                refs_heldout.append(clean_text(ref))
                hyps_raw_heldout.append(clean_text(hyp_raw))
                hyps_denoised_heldout.append(clean_text(hyp_denoised))
                h_surv_raw += s_raw
                h_tot_raw += t_raw
                h_surv_den += s_den
                h_tot_den += t_den
            else:
                refs_indomain.append(clean_text(ref))
                hyps_raw_indomain.append(clean_text(hyp_raw))
                hyps_denoised_indomain.append(clean_text(hyp_denoised))
                i_surv_raw += s_raw
                i_tot_raw += t_raw
                i_surv_den += s_den
                i_tot_den += t_den

        wer_all_raw = jiwer.wer(refs_all, hyps_raw_all) * 100.0
        wer_all_denoised = jiwer.wer(refs_all, hyps_denoised_all) * 100.0

        wer_indomain_raw = jiwer.wer(refs_indomain, hyps_raw_indomain) * 100.0 if refs_indomain else 0.0
        wer_indomain_denoised = jiwer.wer(refs_indomain, hyps_denoised_indomain) * 100.0 if refs_indomain else 0.0

        wer_heldout_raw = jiwer.wer(refs_heldout, hyps_raw_heldout) * 100.0 if refs_heldout else 0.0
        wer_heldout_denoised = jiwer.wer(refs_heldout, hyps_denoised_heldout) * 100.0 if refs_heldout else 0.0

        heldout_num_acc_raw = (h_surv_raw / h_tot_raw * 100.0) if h_tot_raw > 0 else 100.0
        heldout_num_acc_denoised = (h_surv_den / h_tot_den * 100.0) if h_tot_den > 0 else 100.0
        indomain_num_acc_raw = (i_surv_raw / i_tot_raw * 100.0) if i_tot_raw > 0 else 100.0
        indomain_num_acc_denoised = (i_surv_den / i_tot_den * 100.0) if i_tot_den > 0 else 100.0

        summary_rows.append({
            "Condition": cond,
            "Held-Out N": len(refs_heldout),
            "Held-Out Raw WER (%)": round(wer_heldout_raw, 2),
            "Held-Out Denoised WER (%)": round(wer_heldout_denoised, 2),
            "Held-Out Raw Number Acc": f"{h_surv_raw}/{h_tot_raw} ({round(heldout_num_acc_raw, 1)}%)",
            "Held-Out Denoised Number Acc": f"{h_surv_den}/{h_tot_den} ({round(heldout_num_acc_denoised, 1)}%)",
            "In-Domain N": len(refs_indomain),
            "In-Domain Raw WER (%)": round(wer_indomain_raw, 2),
            "In-Domain Denoised WER (%)": round(wer_indomain_denoised, 2),
            "In-Domain Raw Number Acc": f"{i_surv_raw}/{i_tot_raw} ({round(indomain_num_acc_raw, 1)}%)",
        })

    eval_df = pd.DataFrame(summary_rows)
    print("\nEMPIRICAL RESULTS (HELD-OUT VS OVERALL):")
    print(eval_df.to_string(index=False))

    out_csv = PROJECT_ROOT / "benchmark" / "results" / "model_evaluation_metrics.csv"
    eval_df.to_csv(out_csv, index=False)
    print(f"\nSaved held-out evaluation report to: {out_csv}")


if __name__ == "__main__":
    evaluate()
