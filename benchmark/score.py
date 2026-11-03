"""
Saarthi: Benchmark Scoring Harness
Computes official metrics specified in Section 8.2 of the Saarthi Project Guide:
1. Word Error Rate (WER) via jiwer across each noise condition
2. Critical-slot accuracy (% of temperature, distance, location correctly extracted)
3. Errors caught by read-back (The Core Proof: X of Y errors prevented from causing wrong dispatches)
4. Silent failures (Target: 0%)
5. Escalations count
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import jiwer

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_RESULTS_CSV = RESULTS_DIR / "raw_benchmark_results.csv"
SUMMARY_CSV = RESULTS_DIR / "summary_metrics.csv"


def clean_text_for_wer(text: str) -> str:
    """Standardize punctuation and casing for fair WER evaluation."""
    if not text:
        return ""
    import re
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def compute_metrics() -> pd.DataFrame:
    if not RAW_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Raw results not found at {RAW_RESULTS_CSV}. Run run_benchmark.py first.")

    df = pd.read_csv(RAW_RESULTS_CSV)

    summary_rows = []
    conditions = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]

    print("=" * 78)
    print("SAARTHI BENCHMARK EVALUATION (Gnani AI Noisy Telephonic Audio Track)")
    print("=" * 78)

    for cond in conditions:
        sub = df[df["condition"] == cond]
        n_samples = len(sub)
        if n_samples == 0:
            continue

        # 1. Word Error Rate (WER)
        ref_texts = [clean_text_for_wer(t) for t in sub["reference_text"]]
        hyp_texts = [clean_text_for_wer(t) for t in sub["transcript"]]
        wer_val = jiwer.wer(ref_texts, hyp_texts) * 100.0

        # 2. Critical Slot Accuracy (Raw extraction accuracy before read-back)
        total_errors = int(sub["has_extraction_error"].sum())
        correct_extractions = n_samples - total_errors
        slot_acc_raw = (correct_extractions / n_samples) * 100.0

        # 3. Errors Caught by Read-Back (The core claim!)
        errors_caught = int(sub[sub["has_extraction_error"]]["error_caught"].sum())
        caught_pct = (errors_caught / total_errors * 100.0) if total_errors > 0 else 100.0

        # 4. Silent Failures (Critical details wrong that slipped through to dispatch)
        silent_fails = int(sub["silent_failure"].sum())
        silent_fail_pct = (silent_fails / n_samples) * 100.0

        # 5. Post-Confirmation Accuracy (Final accuracy after driver verification)
        post_confirm_acc = ((correct_extractions + errors_caught) / n_samples) * 100.0

        summary_rows.append({
            "Condition": cond,
            "Sample Size (N)": n_samples,
            "WER (%)": round(wer_val, 2),
            "Raw Slot Acc (%)": round(slot_acc_raw, 2),
            "Total ASR Errors": total_errors,
            "Errors Caught by Read-Back": f"{errors_caught} of {total_errors} ({round(caught_pct, 1)}%)",
            "Silent Failures": silent_fails,
            "Final Dispatch Accuracy (%)": round(post_confirm_acc, 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(summary_df.to_string(index=False))
    print("=" * 78)
    print(f"Summary saved to: {SUMMARY_CSV}")
    return summary_df


if __name__ == "__main__":
    compute_metrics()
