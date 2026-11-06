"""
Vaani: Chunked Streaming Trainer & Data Usage Ledger
Implements the 10 GB (or custom size) Chunk -> Train -> Delete -> Log pipeline:
1. Streams/downloads a bounded chunk of speech data (up to target GB/samples)
2. Fine-tunes Whisper-LoRA on GPU
3. Updates saved model adapters in models/whisper_hinglish_lora/
4. Deletes the raw audio chunk from disk immediately
5. Logs data usage to data/training_ledger.json and data/training_ledger.csv
"""

import os
import io
import sys
import json
import time
import shutil
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

LEDGER_JSON = BASE_DIR / "data" / "training_ledger.json"
LEDGER_CSV = BASE_DIR / "data" / "training_ledger.csv"
MODEL_DIR = BASE_DIR / "models" / "whisper_hinglish_lora"
TEMP_CHUNK_DIR = BASE_DIR / "temp_chunk"

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, LoraConfig, get_peft_model


class ChunkAudioDataset(Dataset):
    """Dataset loaded from staged chunk files."""
    def __init__(self, samples: List[Dict[str, Any]], processor: WhisperProcessor):
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        data, sr = sf.read(item["audio_path"])

        if sr != 16000:
            data = signal.resample_poly(data, 16000, sr)

        data = data.astype(np.float32)
        target_len = 16000 * 30
        if len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)))
        else:
            data = data[:target_len]

        input_features = self.processor.feature_extractor(
            data,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(item["text"]).input_ids

        return {
            "input_features": input_features,
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def collate_chunk_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_features = torch.stack([b["input_features"] for b in batch])
    labels_list = [b["labels"] for b in batch]
    max_label_len = max(len(l) for l in labels_list)
    padded_labels = torch.full((len(labels_list), max_label_len), -100, dtype=torch.long)
    for i, l in enumerate(labels_list):
        padded_labels[i, :len(l)] = l
    return {
        "input_features": input_features,
        "labels": padded_labels
    }


def load_ledger() -> Dict[str, Any]:
    if LEDGER_JSON.exists():
        try:
            with open(LEDGER_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"cumulative_gb_processed": 0.0, "cumulative_hours_processed": 0.0, "cumulative_samples_trained": 0, "history": []}


def save_ledger(ledger: Dict[str, Any]):
    LEDGER_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_JSON, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)

    # Export CSV summary
    rows = []
    for h in ledger.get("history", []):
        rows.append({
            "Chunk ID": h.get("chunk_id"),
            "Dataset Source": h.get("source"),
            "Samples": h.get("samples"),
            "Size (GB)": h.get("size_gb"),
            "Duration (Hours)": h.get("hours"),
            "Loss": h.get("post_loss"),
            "Status": h.get("status"),
            "Timestamp": h.get("timestamp")
        })
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER_CSV, index=False)


def print_ledger_summary():
    ledger = load_ledger()
    print("\n" + "=" * 80)
    print("📊 VAANI DATASET TRAINING USAGE LEDGER")
    print("=" * 80)
    print(f"Total Data Processed : {ledger.get('cumulative_gb_processed', 0.0):.2f} GB")
    print(f"Total Audio Duration : {ledger.get('cumulative_hours_processed', 0.0):.2f} Hours")
    print(f"Total Utterances     : {ledger.get('cumulative_samples_trained', 0):,}")
    print("-" * 80)

    for i, h in enumerate(ledger.get("history", []), 1):
        print(f"[{i:02d}] {h['chunk_id']:<20} | {h['size_gb']:>5.2f} GB | {h['hours']:>5.2f} hrs | {h['samples']:>5} clips | Loss: {h.get('post_loss', 'N/A')} | {h.get('source', '')[:35]}")
    print("=" * 80 + "\n")


def stage_and_train_chunk(
    dataset_name: str = "dianavdavidson/gram-vaani-preprocessed",
    dataset_config: Optional[str] = None,
    chunk_id: str = "chunk_01_gramvaani",
    max_samples: int = 400,
    max_gb: float = 10.0,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 3e-4
) -> Dict[str, Any]:
    """
    1. Downloads/streams up to max_samples or max_gb into TEMP_CHUNK_DIR
    2. Measures volume & audio hours
    3. Trains Whisper-LoRA on GPU
    4. Deletes TEMP_CHUNK_DIR
    5. Records entry in persistent ledger
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=================================================================")
    print(f"STARTING CHUNKED TRAINING CYCLE: {chunk_id}")
    print(f"Source Dataset: {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
    print(f"Max Samples: {max_samples} | Max Chunk Limit: {max_gb:.1f} GB")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"=================================================================\n")

    TEMP_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))

    # --- Phase 1: Stream and Stage Audio Chunk ---
    print(f"📥 [Phase 1/4] Streaming and staging audio chunk into temporary cache...")
    from datasets import load_dataset, Audio

    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True).cast_column("audio", Audio(decode=False))
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True).cast_column("audio", Audio(decode=False))

    staged_samples: List[Dict[str, Any]] = []
    total_bytes = 0
    total_audio_sec = 0.0

    t_start = time.time()
    for i, item in enumerate(ds):
        if len(staged_samples) >= max_samples:
            break

        audio_bytes = item["audio"]["bytes"]
        # Extract text field (sentence, transcription, or text)
        text = item.get("sentence") or item.get("transcription") or item.get("text") or item.get("transcript") or ""
        text = str(text).strip()

        if not text or len(audio_bytes) < 1000:
            continue

        # Save to temp wav
        wav_path = TEMP_CHUNK_DIR / f"clip_{len(staged_samples):05d}.wav"
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        # Measure audio duration
        try:
            info = sf.info(str(wav_path))
            total_audio_sec += info.duration
        except Exception:
            total_audio_sec += 4.0

        total_bytes += len(audio_bytes)
        staged_samples.append({
            "audio_path": str(wav_path),
            "text": text
        })

        # Check GB limit
        cur_gb = total_bytes / (1024 ** 3)
        if cur_gb >= max_gb:
            print(f"Reached chunk size limit: {cur_gb:.2f} GB")
            break

        if len(staged_samples) % 25 == 0:
            print(f"  Staged {len(staged_samples)}/{max_samples} clips ({cur_gb * 1024:.1f} MB, {total_audio_sec/60:.1f} mins audio)...", flush=True)

    chunk_size_gb = round(total_bytes / (1024 ** 3), 4)
    chunk_hours = round(total_audio_sec / 3600.0, 3)
    print(f"✅ Staged {len(staged_samples)} clips | Size: {chunk_size_gb * 1024:.1f} MB ({chunk_size_gb:.3f} GB) | Duration: {chunk_hours:.2f} hours", flush=True)

    # --- Phase 2: Load Model and Train on Chunk ---
    print(f"\n🧠 [Phase 2/4] Training Whisper-LoRA on Chunk ({epochs} epochs on GPU)...")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model = PeftModel.from_pretrained(base_model, str(MODEL_DIR), is_trainable=True)
    model.to(device)
    model.train()

    dataset = ChunkAudioDataset(staged_samples, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_chunk_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, batches)
        final_loss = avg_loss
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

    # Save updated adapters
    print(f"💾 Saving updated LoRA adapters to {MODEL_DIR}...", flush=True)
    model.save_pretrained(str(MODEL_DIR))
    processor.save_pretrained(str(MODEL_DIR))

    # --- Phase 3: Purge & Delete Raw Audio Chunk ---
    print(f"\n🗑️ [Phase 3/4] Purging temporary raw audio data ({chunk_size_gb * 1024:.1f} MB)...", flush=True)
    del model
    del base_model
    del dataset
    del dataloader
    torch.cuda.empty_cache()

    try:
        shutil.rmtree(TEMP_CHUNK_DIR)
        print(f"✅ Temporary chunk folder {TEMP_CHUNK_DIR} successfully DELETED from disk.")
    except Exception as e:
        print(f"Warning deleting temp chunk: {e}")

    # --- Phase 4: Record in Persistent Ledger ---
    print(f"\n📝 [Phase 4/4] Logging entry to persistent Training Ledger...")
    ledger = load_ledger()
    ledger["cumulative_gb_processed"] = round(ledger.get("cumulative_gb_processed", 0.0) + chunk_size_gb, 4)
    ledger["cumulative_hours_processed"] = round(ledger.get("cumulative_hours_processed", 0.0) + chunk_hours, 3)
    ledger["cumulative_samples_trained"] = ledger.get("cumulative_samples_trained", 0) + len(staged_samples)

    chunk_record = {
      "chunk_id": chunk_id,
      "source": f"{dataset_name} ({dataset_config})" if dataset_config else dataset_name,
      "samples": len(staged_samples),
      "size_gb": chunk_size_gb,
      "hours": chunk_hours,
      "post_loss": round(final_loss, 4),
      "status": "COMPLETED & PURGED",
      "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    ledger["history"].append(chunk_record)
    save_ledger(ledger)

    print_ledger_summary()
    return chunk_record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dianavdavidson/gram-vaani-preprocessed")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--chunk_id", type=str, default="chunk_01_gramvaani")
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--max_gb", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        print_ledger_summary()
    else:
        stage_and_train_chunk(
            dataset_name=args.dataset,
            dataset_config=args.config,
            chunk_id=args.chunk_id,
            max_samples=args.samples,
            max_gb=args.max_gb,
            epochs=args.epochs
        )
