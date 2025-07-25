import librosa
import numpy as np
import colorsys

from config import VisualConfig

def short_time_fourrier_transform(audio_file: str, config: VisualConfig) -> np.ndarray:
    """
    Load audio file and compute its Short Time Fourier Transform (STFT).
    :param audio_file: Path to the audio file.
    :param config: VisualConfig object with settings.
    :return: STFT of the audio file, shape [freq_bins, frames].
    """
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    hop_length = int(sr / config.fps)  # frames per second
    bands = config.num_frequency_bands
    frames = config.duration * config.fps
    stft = np.abs(librosa.stft(y, n_fft=bands * 2, hop_length=hop_length))
    stft = stft[:bands, :frames]
    stft = stft / np.max(stft) # normalize
    return stft

class AudioInfo:
    """
    Class to hold audio information for a specific frame.
    """
    def __init__(self, loudness: float, avg_freq: float, color: tuple):
        self.loudness = loudness
        self.avg_freq = avg_freq
        self.color = color

def frequency_to_color(ratio: float) -> tuple:
    """
    Convert frequency to RGB color.
    :param ratio: Normalized frequency ratio in [0, 1].
    :return: RGB color tuple.
    """
    ratio = np.clip(ratio, 0, 1)
    hue = ratio  # hue in [0, 1]
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    return (r, g, b)

def get_audio_info(stft: np.ndarray, config: VisualConfig) -> list:
    """
    Compute AudioInfo for all frames, with normalization.
    :param stft: Short Time Fourier Transform of the audio, shape [freq_bins, frames].
    :param config: VisualConfig object with settings.
    :return: List of AudioInfo objects for each frame."""
    freqs = np.linspace(0, config.num_frequency_bands // 2, stft.shape[0])
    max_freq = freqs[-1]
    freq_weights = 1.0 - (freqs / max_freq) ** config.freq_band_weight_func_exponent
    loudness_arr = np.sum(stft * freq_weights[:, None], axis=0)
    avg_freq_arr = np.sum(freqs[:, None] * stft, axis=0) / (np.sum(stft, axis=0) + 1e-8)

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
