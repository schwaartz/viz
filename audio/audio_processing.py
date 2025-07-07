import librosa
import numpy as np
from constants import AUDIO_FILE, FPS, DURATION, NUM_FREQ
import colorsys

def short_time_fourrier_transform() -> np.ndarray:
    """Load audio file and compute its Short Time Fourier Transform (STFT)."""
    y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
    hop_length = int(sr / FPS)
    stft = np.abs(librosa.stft(y, n_fft=NUM_FREQ * 2, hop_length=hop_length))
    stft = stft[:NUM_FREQ, :DURATION * FPS]  # [freq_bins, frames]
    stft = stft / np.max(stft)  # normalize
    return stft

class AudioInfo:
    """Class to hold audio information for a specific frame."""
    def __init__(self, loudness, avg_freq, color):
        self.loudness = loudness
        self.avg_freq = avg_freq
        self.color = color

def frequency_to_color(ratio: float) -> tuple:
    """Convert frequency to RGB color."""
    ratio = np.clip(ratio, 0, 1)
    hue = ratio  # hue in [0, 1]
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    return (r, g, b)

def get_audio_info(stft: np.ndarray, sr: int) -> list:
    """Compute AudioInfo for all frames, with normalization."""
    freqs = np.linspace(0, sr // 2, stft.shape[0])
    loudness_arr = np.sum(stft, axis=0)
    avg_freq_arr = np.sum(freqs[:, None] * stft, axis=0) / (loudness_arr + 1e-8)

    # Normalize loudness and avg_freq
    loudness_min, loudness_max = loudness_arr.min(), loudness_arr.max()
    avg_freq_min, avg_freq_max = avg_freq_arr.min(), avg_freq_arr.max()
    loudness_norm = (loudness_arr - loudness_min) / (loudness_max - loudness_min + 1e-8)
    avg_freq_norm = (avg_freq_arr - avg_freq_min) / (avg_freq_max - avg_freq_min + 1e-8)

    infos = []
    for i in range(stft.shape[1]):
        color = frequency_to_color(avg_freq_norm[i])
        infos.append(AudioInfo(loudness_norm[i], avg_freq_norm[i], color))
    return infos
