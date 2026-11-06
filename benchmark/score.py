"""
Vaani: Empirical Benchmark Scoring Module
Computes:
1. Word Error Rate (WER) using jiwer
2. Raw Critical Slot Extraction Accuracy (before human read-back)
3. Errors Caught by Spoken Read-Back Gate
4. Silent Failures (Uncaught errors slipping through to dispatch)
5. True Final Dispatch Accuracy (Non-circular calculation)
"""

import re
from pathlib import Path
import pandas as pd
import jiwer

RAW_RESULTS_CSV = Path(__file__).resolve().parent / "results" / "raw_benchmark_results.csv"
SUMMARY_CSV = Path(__file__).resolve().parent / "results" / "summary_metrics.csv"


def clean_text_for_wer(text: str) -> str:
    """Normalize text for consistent WER scoring."""
    t = str(text).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def compute_metrics() -> pd.DataFrame:
    if not RAW_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Raw results not found at: {RAW_RESULTS_CSV}")

    df = pd.read_csv(RAW_RESULTS_CSV)

    summary_rows = []
    conditions = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]

    print("=" * 80)
    print("VAANI EMPIRICAL BENCHMARK EVALUATION METRICS")
    print("=" * 80)

    for cond in conditions:
        sub = df[df["condition"] == cond]
        n_samples = len(sub)
        if n_samples == 0:
            continue

        # 1. Word Error Rate (WER)
        ref_texts = [clean_text_for_wer(t) for t in sub["reference_text"]]
        hyp_texts = [clean_text_for_wer(t) for t in sub["transcript"]]
        wer_val = jiwer.wer(ref_texts, hyp_texts) * 100.0

        # 2. Raw Critical Slot Accuracy (Before Read-Back)
        total_errors = int(sub["has_extraction_error"].sum())
        correct_extractions = n_samples - total_errors
        slot_acc_raw = (correct_extractions / n_samples) * 100.0

        # 3. Errors Caught by Read-Back
        errors_caught = int(sub[sub["has_extraction_error"]]["error_caught"].sum())
        caught_pct = (errors_caught / total_errors * 100.0) if total_errors > 0 else 100.0

        # 4. Silent Failures (Driver misheard / reflexively confirmed incorrect details)
        silent_fails = int(sub["silent_failure"].sum())
        silent_fail_pct = (silent_fails / n_samples) * 100.0

        # 5. Non-Circular True Dispatch Accuracy
        # Dispatch is accurate if raw extraction was correct OR error was caught & corrected
        true_dispatch_acc = ((n_samples - silent_fails) / n_samples) * 100.0

        summary_rows.append({
            "Condition": cond,
            "Sample Size (N)": n_samples,
            "WER (%)": round(wer_val, 2),
            "Raw Slot Acc (%)": round(slot_acc_raw, 2),
            "Total Errors": total_errors,
            "Errors Caught": f"{errors_caught} of {total_errors} ({round(caught_pct, 1)}%)",
            "Silent Failures": f"{silent_fails} ({round(silent_fail_pct, 1)}%)",
            "Final Dispatch Accuracy (%)": round(true_dispatch_acc, 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"Summary metrics saved to: {SUMMARY_CSV}")
    return summary_df


if __name__ == "__main__":
    compute_metrics()
