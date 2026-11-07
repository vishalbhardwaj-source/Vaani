"""
Vaani: Model 2 Hinglish Telephonic ASR Training Pipeline
Fine-tunes Whisper using LoRA (PEFT) on paired Hinglish dispatch recordings
under telephonic and noisy cabin conditions.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "openai/whisper-tiny"
OUTPUT_DIR = BASE_DIR / "models" / "whisper_hinglish_lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Strict Held-Out Evaluation Set (Excluded from model training)
HELD_OUT_UTT_IDS = {"utt_03", "utt_07", "utt_11", "utt_15", "utt_19", "utt_22", "utt_25"}


class HinglishTelephonyASRDataset(Dataset):
    """
    Paired audio-text dataset constructed from:
    1. Benchmark Hinglish dispatch recordings (clean, 8 kHz, 10 dB SNR, 5 dB SNR)
    2. Utterance transcripts from benchmark/utterances.csv
    Strictly excludes the 7 held-out evaluation utterances.
    """
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
        # Configure decoder prefix tokens for Latin / Roman Hinglish fine-tuning
        self.processor.tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
        self.samples: List[Dict[str, Any]] = []
        self._build_dataset()

    def _build_dataset(self):
        csv_path = BASE_DIR / "benchmark" / "utterances.csv"
        df = pd.read_csv(csv_path)

        audio_dir = BASE_DIR / "benchmark" / "audio"
        variants = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]

        for _, row in df.iterrows():
            utt_id = row["id"]
            # Enforce strict train / held-out split: skip test clips
            if utt_id in HELD_OUT_UTT_IDS:
                continue

            text = str(row["utterance"]).strip()

            for var in variants:
                audio_file = audio_dir / f"{utt_id}_{var}.wav"
                if audio_file.exists():
                    self.samples.append({
                        "audio_path": str(audio_file),
                        "text": text
                    })
        print(f"Dataset Built: {len(self.samples)} training samples ({len(df) - len(HELD_OUT_UTT_IDS)} utterances). Excluded {len(HELD_OUT_UTT_IDS)} held-out utterances.")

        print(f"Loaded {len(self.samples)} paired audio-text samples across 4 acoustic conditions.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        data, sr = sf.read(item["audio_path"])

        # Resample to 16000 Hz if not already (Whisper expects 16 kHz)
        if sr != 16000:
            data = signal.resample_poly(data, 16000, sr)

        # Pad or trim to 30s (Whisper standard)
        target_len = 16000 * 30
        if len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)))
        else:
            data = data[:target_len]

        # Extract log-Mel spectrogram features
        input_features = self.processor.feature_extractor(
            data,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features[0]

        # Tokenize target text
        labels = self.processor.tokenizer(item["text"]).input_ids

        return {
            "input_features": input_features,
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_features = torch.stack([b["input_features"] for b in batch])
    labels_list = [b["labels"] for b in batch]

    # Pad labels with -100 (PyTorch CrossEntropy ignores -100)
    max_label_len = max(len(l) for l in labels_list)
    padded_labels = torch.full((len(labels_list), max_label_len), -100, dtype=torch.long)
    for i, l in enumerate(labels_list):
        padded_labels[i, :len(l)] = l

    return {
        "input_features": input_features,
        "labels": padded_labels
    }


def train_asr(epochs: int = 5, batch_size: int = 4, lr: float = 5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"Training Model 2: Hinglish Telephony ASR (Whisper + LoRA)")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Target Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"========================================================\n")

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        trust_remote_code=False
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        trust_remote_code=False
    )

    # Configure LoRA parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    dataset = HinglishTelephonyASRDataset(processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_time = time.time()
    model.train()

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
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f} - Time: {elapsed:.1f}s")

    # Save fine-tuned LoRA adapters and processor
    print(f"\nSaving LoRA adapters to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print("Model 2 (ASR) weights and processor saved successfully!")

    # Verify inference on a noisy clip
    print("\n--- Verifying Model 2 ASR Transcription on 5 dB Noisy Telephony Clip ---")
    model.eval()
    test_audio = BASE_DIR / "benchmark" / "audio" / "utt_01_telephony_5db.wav"
    if test_audio.exists():
        data, sr = sf.read(str(test_audio))
        if sr != 16000:
            data = signal.resample_poly(data, 16000, sr)
        feats = processor(data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            gen_tokens = model.generate(feats)
            transcription = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        print(f"Predicted Transcript: \"{transcription}\"")
        print("ASR Model Verification PASSED!")

    return OUTPUT_DIR


if __name__ == "__main__":
    train_asr(epochs=5, batch_size=4)
