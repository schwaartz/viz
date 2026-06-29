import librosa
import numpy as np

def generate_spectrogram(
    audio_file: str,
    freq: float,
    start_time: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    """
    Load audio file and compute its Short Time Fourier Transform (STFT).
    :param audio_file: Path to the audio file.
    :param freq: Frequency in Hz.
    :return: Normalized spectrogram as a 2D numpy array.
    """
    
    y, sr = librosa.load(audio_file, sr=None, mono=True, offset=start_time, duration=duration)
    hop_length = int(sr / freq) # Spectrums per second
    bands = 128  # Number of frequency bands
    window_duration = duration if duration is not None else librosa.get_duration(y=y, sr=sr)
    spectrums = int(window_duration * freq)  # Total number of spectrums
    stft = librosa.stft(y, n_fft=bands * 2, hop_length=hop_length)
    mag = np.abs(stft)  # Amplitude (real-valued)

    # Trim or pad time axis so output has exactly `spectrums` columns
    mag = mag[:bands, :]
    if mag.shape[1] < spectrums:
        pad = np.zeros((bands, spectrums - mag.shape[1]), dtype=mag.dtype)
        mag = np.concatenate([mag, pad], axis=1)
    else:
        mag = mag[:, :spectrums]

    # Convert amplitude to dB (log scale) and normalize to [0,1]
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)
    mag_db -= mag_db.min()
    mag_db /= (mag_db.max() + 1e-8)

    return mag_db.astype(np.float32)
