"""
Vaani: Unified 3-Stage Trained Model Pipeline
Runs end-to-end inference across all 3 trained models on RTX 5070 GPU:
1. Model 1 (Denoiser): models/denoiser_8khz.pt
2. Model 2 (ASR): models/whisper_hinglish_lora/
3. Model 3 (NLU): models/nlu_slot_extractor/
"""

import os
import sys
import json
import torch
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.denoiser_model import TelephonyDenoiser
from core.asr_model import get_asr_service
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


class TrainedVaaniPipeline:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Initializing Vaani 3-Stage Pipeline on {self.device}...")

        # 1. Load Model 1: Denoiser
        self.denoiser = TelephonyDenoiser()
        denoiser_path = BASE_DIR / "models" / "denoiser_8khz.pt"
        if denoiser_path.exists():
            self.denoiser.load_state_dict(torch.load(str(denoiser_path), map_location=self.device, weights_only=True))
        self.denoiser.to(self.device).eval()
        print("Model 1 (8 kHz Denoiser) loaded.")

        # 2. Load Model 2: Hinglish ASR via singleton
        self.asr_service = get_asr_service()
        self.asr_processor = self.asr_service.processor
        self.asr_model = self.asr_service.model
        print("Model 2 (Hinglish ASR Whisper+LoRA) loaded.")

        # 3. Load Model 3: NLU Slot Extractor
        nlu_dir = BASE_DIR / "models" / "nlu_slot_extractor"
        self.nlu_tokenizer = AutoTokenizer.from_pretrained(str(nlu_dir))
        base_nlu = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small",
            revision="0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
            trust_remote_code=False
        )
        self.nlu_model = PeftModel.from_pretrained(base_nlu, str(nlu_dir))
        self.nlu_model.to(self.device).eval()
        print("Model 3 (NLU Slot Extractor Flan-T5+LoRA) loaded.")

    def process(self, audio_path: str) -> Dict[str, Any]:
        """
        Execute full pipeline:
        Noisy Audio -> Denoise (Model 1) -> Transcribe (Model 2) -> Extract Slots (Model 3)
        """
        # Step 1: Denoise
        data, sr = sf.read(audio_path)
        if sr != 8000:
            data = signal.resample_poly(data, 8000, sr)

        noisy_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
        cleaned_tensor = self.denoiser.enhance(noisy_tensor)
        cleaned_np = cleaned_tensor.cpu().numpy()

        # Step 2: Transcribe via Model 2
        # Whisper expects 16 kHz
        cleaned_16k = signal.resample_poly(cleaned_np, 16000, 8000)
        feats = self.asr_processor(cleaned_16k, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with torch.no_grad():
            tokens = self.asr_model.generate(feats)
            transcript = self.asr_processor.batch_decode(tokens, skip_special_tokens=True)[0]

        # Step 3: Extract Slots via Model 3
        prompt = f"extract dispatch slots: {transcript}"
        in_tokens = self.nlu_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.nlu_model.generate(**in_tokens, max_length=128)
            nlu_output = self.nlu_tokenizer.decode(out_ids[0], skip_special_tokens=True)

        return {
            "cleaned_samples": len(cleaned_np),
            "transcript": transcript,
            "nlu_raw": nlu_output
        }


if __name__ == "__main__":
    pipeline = TrainedVaaniPipeline()
    test_audio = str(BASE_DIR / "benchmark" / "audio" / "utt_01_telephony_5db.wav")
    print(f"\nRunning end-to-end test on severe 5 dB noisy telephonic audio: {test_audio}")
    result = pipeline.process(test_audio)
    print("\n--- Pipeline Execution Output ---")
    print("Transcribed Speech:", result["transcript"])
    print("NLU Output:", result["nlu_raw"])
    print("All 3 models executed in sequence successfully!")
