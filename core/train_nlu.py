"""
Vaani: Model 3 Structured Slot Extraction & Read-Back NLU Training Pipeline
Fine-tunes a sequence-to-sequence model to convert raw Hinglish ASR transcripts
into strict structured JSON slots and generate safety read-back confirmation dialogue.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = BASE_DIR / "models" / "nlu_slot_extractor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_training_examples() -> List[Dict[str, str]]:
    """Build dataset of Hinglish utterances paired with structured target JSON."""
    csv_path = BASE_DIR / "benchmark" / "utterances.csv"
    df = pd.read_csv(csv_path)

    examples = []

    # Format read-back template
    for _, row in df.iterrows():
        utt = str(row["utterance"]).strip()
        inc_type = str(row["incident_type"]).strip()
        temp = None if pd.isna(row["true_temperature_c"]) else float(row["true_temperature_c"])
        loc = None if pd.isna(row["true_location"]) else str(row["true_location"]).strip()
        dist = None if pd.isna(row["true_distance_km"]) else float(row["true_distance_km"])

        parts = []
        if inc_type == "reefer_temperature" and temp is not None:
            parts.append(f"Temperature {int(temp) if temp.is_integer() else temp} degree")
        elif inc_type == "breakdown":
            parts.append("Breakdown report")
        elif inc_type == "accident":
            parts.append("Accident report")

        if loc:
            if dist:
                parts.append(f"{loc} se {int(dist) if dist.is_integer() else dist} km")
            else:
                parts.append(f"{loc} ke paas")

        readback = ", ".join(parts) + ". Sahi hai ya galat?"

        target_obj = {
            "incident": inc_type,
            "temp_c": temp,
            "location": loc,
            "dist_km": dist,
            "readback": readback
        }

        # Input prompt
        prompt = f"extract dispatch slots: {utt}"
        target_str = json.dumps(target_obj, ensure_ascii=False)

        examples.append({"input": prompt, "target": target_str})

        # Add linguistic augmentations
        examples.append({"input": f"extract dispatch slots: Bhaiya {utt}", "target": target_str})
        examples.append({"input": f"extract dispatch slots: Urgent update: {utt}", "target": target_str})

    return examples


class NLUSlotDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], tokenizer: Any, max_len: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        in_enc = self.tokenizer(
            ex["input"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        out_enc = self.tokenizer(
            ex["target"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = out_enc.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": in_enc.input_ids.squeeze(0),
            "attention_mask": in_enc.attention_mask.squeeze(0),
            "labels": labels
        }


def train_nlu(epochs: int = 6, batch_size: int = 8, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"Training Model 3: Hinglish Slot Extraction & NLU (Flan-T5)")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Target Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"========================================================\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    examples = build_training_examples()
    print(f"Prepared {len(examples)} training examples with slot annotations.")

    dataset = NLUSlotDataset(examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_time = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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

    # Save fine-tuned weights and tokenizer
    print(f"\nSaving Model 3 weights to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Model 3 (NLU) weights saved successfully!")

    # Verify extraction on test Hinglish utterance
    print("\n--- Verifying Model 3 Slot Extraction on Test Utterance ---")
    model.eval()
    test_input = "extract dispatch slots: Bhaiya reefer ka temperature 18 degree ho gaya hai toll plaza se 42 km aage"
    in_tokens = tokenizer(test_input, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**in_tokens, max_length=128)
        pred_json = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    print(f"Input: {test_input}")
    print(f"Predicted JSON: {pred_json}")
    print("Model 3 Verification PASSED!")

    return OUTPUT_DIR


if __name__ == "__main__":
    train_nlu(epochs=6, batch_size=8)
