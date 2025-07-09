import librosa
import numpy as np
from constants import *
import colorsys

def short_time_fourrier_transform() -> np.ndarray:
    """
    Load audio file and compute its Short Time Fourier Transform (STFT).
    :return: STFT of the audio file, shape [freq_bins, frames].
    """
    y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
    hop_length = int(sr / FPS)
    stft = np.abs(librosa.stft(y, n_fft=NUM_FREQ * 2, hop_length=hop_length))
    stft = stft[:NUM_FREQ, :DURATION * FPS]  # [freq_bins, frames]
    stft = stft / np.max(stft)  # normalize
    return stft

class AudioInfo:
    """
    Class to hold audio information for a specific frame.
    """
    def __init__(self, loudness: float, avg_freq: float, color: tuple, protrusions: list):
        self.loudness = loudness
        self.avg_freq = avg_freq
        self.color = color
        self.protrusions = protrusions

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

def get_protrusion_array(stft: np.ndarray) -> np.ndarray:
    """
    Compute protrusion array based on the STFT.
    The protrusions are caculated by dividing the STFT values into MAX_ACTIVE_PROTRUSIONS bands of frequencies
    and then calculating the loudness for each band. Then, at the end, the size for each protrusion band is normalized.
    :param stft: Short Time Fourier Transform of the audio, shape [freq_bins, frames].
    :return: Protrusion array, shape [frames, MAX_ACTIVE_PROTRUSIONS].
    """
    protrusion_arr = np.zeros((stft.shape[1], MAX_ACTIVE_PROTRUSIONS))
    for i in range(stft.shape[1]):
        band_size = stft.shape[0] // MAX_ACTIVE_PROTRUSIONS
        for j in range(MAX_ACTIVE_PROTRUSIONS):
            start = j * band_size
            end = (j + 1) * band_size if j <= MAX_ACTIVE_PROTRUSIONS else stft.shape[0]
            protrusion_arr[i, j] = np.sum(stft[start:end, i])

    # Normalize protrusions
    min_arr = protrusion_arr.min(axis=0) # Sum over all the frames
    max_arr = protrusion_arr.max(axis=0) # Sum over all the frames
    protrusion_arr = (protrusion_arr - min_arr) / (max_arr - min_arr + 1e-8)
    protrusion_arr = protrusion_arr * PORTR_SCALE  # Scale protrusions
    return protrusion_arr
        


def get_audio_info(stft: np.ndarray, sr: int) -> list:
    """
    Compute AudioInfo for all frames, with normalization.
    :param stft: Short Time Fourier Transform of the audio, shape [freq_bins, frames].
    :param sr: Sample rate of the audio.
    :return: List of AudioInfo objects for each frame."""
    freqs = np.linspace(0, sr // 2, stft.shape[0])
    max_freq = freqs[-1]
    freq_weights = 1.0 - (freqs / max_freq) ** LOWER_FREQ_WEIGHT_FUNC_EXPONENT
    loudness_arr = np.sum(stft * freq_weights[:, None], axis=0)
    avg_freq_arr = np.sum(freqs[:, None] * stft, axis=0) / (np.sum(stft, axis=0) + 1e-8)
    portr_arr = get_protrusion_array(stft)

    # Normalize loudness and avg_freq
    loudness_min, loudness_max = loudness_arr.min(), loudness_arr.max()
    avg_freq_min, avg_freq_max = avg_freq_arr.min(), avg_freq_arr.max()
    loudness_norm = (loudness_arr - loudness_min) / (loudness_max - loudness_min + 1e-8)
    avg_freq_norm = (avg_freq_arr - avg_freq_min) / (avg_freq_max - avg_freq_min + 1e-8)

    infos = []
    for i in range(stft.shape[1]):
        color = frequency_to_color(avg_freq_norm[i])
        prortr_list = [[MIN_PROTRUSIONS + j, portr_arr[i, j]] for j in range(MAX_ACTIVE_PROTRUSIONS)]
        infos.append(AudioInfo(loudness_norm[i], avg_freq_norm[i], color, prortr_list))
    return infos
