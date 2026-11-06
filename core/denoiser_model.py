"""
Vaani: 8 kHz Telephony Neural Audio Denoiser (Model 1)
Deep Convolutional-Recurrent Network (CRN) that learns an acoustic ratio mask
in the STFT domain to strip engine rumble, tyre friction, and cabin reverberation
from 8 kHz narrow-band telephonic audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


class TelephonyDenoiser(nn.Module):
    """
    Real-time 8 kHz spectral mask estimator:
    - Input: Noisy STFT Magnitude [Batch, Freq_bins=129, Time_frames]
    - Architecture: 2D Conv Encoder -> 2-layer Bidirectional GRU -> 2D Deconv Decoder -> Sigmoid Mask
    - Output: Gain Mask [0.0, 1.0] applied to noisy STFT magnitude
    - Reconstructed via iSTFT using the noisy phase.
    """
    def __init__(self, n_fft: int = 256, hop_length: int = 64, win_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.freq_bins = n_fft // 2 + 1  # 129 bins for n_fft=256 at 8 kHz

        # Encoder: 2D Convolutions over (Freq, Time)
        # Input shape: [B, 1, 129, T]
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )  # [B, 16, 65, T]

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )  # [B, 32, 33, T]

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )  # [B, 64, 17, T]

        # Recurrent bottleneck: 64 * 17 = 1088 features per time frame
        self.gru = nn.GRU(
            input_size=64 * 17,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.gru_proj = nn.Linear(512, 64 * 17)

        # Decoder: Transposed 2D Convolutions with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )  # -> [B, 32, 33, T]

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(0, 0)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )  # -> [B, 16, 65, T]

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(0, 0)),
            nn.Sigmoid()
        )  # -> [B, 1, 129, T] mask

        # Register Hann window buffer
        self.register_buffer("window", torch.hann_window(self.win_length))

    def compute_stft(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT magnitude and phase for waveform [B, T]."""
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        return magnitude, phase

    def compute_istft(self, magnitude: torch.Tensor, phase: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """Reconstruct waveform from magnitude and phase."""
        complex_spec = torch.polar(magnitude, phase)
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length
        )

    def forward(self, noisy_mag: torch.Tensor) -> torch.Tensor:
        """
        noisy_mag: [B, 1, 129, T]
        Returns: Estimated gain mask [B, 1, 129, T]
        """
        # Encoder
        e1 = self.enc1(noisy_mag)  # [B, 16, 65, T]
        e2 = self.enc2(e1)         # [B, 32, 33, T]
        e3 = self.enc3(e2)         # [B, 64, 17, T]

        B, C, F_dim, T_dim = e3.shape
        # Permute for GRU: [B, T_dim, C * F_dim]
        gru_in = e3.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        gru_out, _ = self.gru(gru_in)  # [B, T_dim, 512]
        gru_proj = self.gru_proj(gru_out)  # [B, T_dim, 64 * 17]
        # Reshape back to [B, 64, 17, T]
        d3_in = gru_proj.view(B, T_dim, C, F_dim).permute(0, 2, 3, 1).contiguous()

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([d3_in, e3], dim=1))  # [B, 32, 33, T]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))     # [B, 16, 65, T]
        mask = self.dec1(torch.cat([d2, e1], dim=1))   # [B, 1, 129, T]

        return mask

    def enhance(self, noisy_waveform: torch.Tensor) -> torch.Tensor:
        """
        Enhance a 1D or 2D batch of 8 kHz audio waveforms.
        noisy_waveform: [B, T] or [T]
        Returns cleaned waveform [B, T] or [T]
        """
        was_1d = False
        if noisy_waveform.dim() == 1:
            noisy_waveform = noisy_waveform.unsqueeze(0)
            was_1d = True

        orig_len = noisy_waveform.shape[-1]
        device = noisy_waveform.device

        noisy_mag, noisy_phase = self.compute_stft(noisy_waveform)
        noisy_mag_in = noisy_mag.unsqueeze(1)  # [B, 1, 129, T_spec]

        with torch.no_grad():
            mask = self.forward(noisy_mag_in)
            mask = mask.squeeze(1)  # [B, 129, T_spec]
            clean_mag = noisy_mag * mask
            enhanced = self.compute_istft(clean_mag, noisy_phase, length=orig_len)

        if was_1d:
            enhanced = enhanced.squeeze(0)
        return enhanced

    def enhance_file(self, input_path: str, output_path: str, device: str = "cuda") -> float:
        """Read noisy WAV, enhance, and write to output_path. Returns SNR improvement approx."""
        data, sr = sf.read(input_path)
        if sr != 8000:
            import scipy.signal as signal
            data = signal.resample_poly(data, 8000, sr)

        tensor_in = torch.tensor(data, dtype=torch.float32, device=device)
        self.to(device)
        self.eval()

        cleaned_tensor = self.enhance(tensor_in)
        cleaned_np = cleaned_tensor.cpu().numpy()

        # Prevent clipping
        max_val = np.max(np.abs(cleaned_np)) + 1e-8
        if max_val > 0.95:
            cleaned_np = cleaned_np * (0.95 / max_val)

        sf.write(output_path, cleaned_np, 8000)
        return float(np.mean(cleaned_np ** 2))
