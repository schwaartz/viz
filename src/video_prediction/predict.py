import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from PIL import Image
import torch
from video_prediction.model import VideoPredictor
from video_prediction.audio_preprocessing import generate_spectrogram
from video_prediction.constants import (
    DEFAULT_MODEL_PATH,
    WINDOW_SECONDS,
    VIDEO_RESIZE,
    VIDEO_TARGET_FPS,
    AUDIO_FEATURES_PER_SECOND
)

def save_video(frames: np.ndarray, output_path: str, fps: int, width: int, height: int):
    """
    Saves the predicted video frames to a video file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(frames)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[1] in (1, 3):
        array = np.transpose(array, (0, 2, 3, 1))

    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).astype(np.uint8)

    # imageio uses an ffmpeg backend here, matching the save flow from the visualizer module.
    writer: Any = imageio.get_writer(str(path), fps=fps)
    try:
        for frame in array:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR))
            writer.append_data(frame)
    finally:
        writer.close()


def _mux_audio_with_video(video_path: str, audio_path: str, output_path: str) -> None:
    """Attach the original audio track to the generated video clip."""
    process = subprocess.Popen([
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        output_path,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()
    if process.returncode != 0:
        raise RuntimeError("ffmpeg failed while muxing audio and video")


def _iter_audio_window_starts(audio_path: str, window_seconds: float) -> list[float]:
    """Return non-overlapping window starts that cover the whole audio file."""
    import librosa

    duration = float(librosa.get_duration(path=audio_path))
    if duration <= 0:
        return [0.0]

    starts = []
    start = 0.0
    while start < duration:
        starts.append(round(start, 6))
        start += window_seconds
    return starts or [0.0]


def main():
    """
    Main function to predict a full video from a given audio file using the
    trained model. It loads the model, processes the audio, and generates the
    predicted video frames.
    """
    parser = argparse.ArgumentParser(description="Predict video from audio")
    parser.add_argument("--input-audio", "-i", type=str, required=True)
    parser.add_argument("--output-video", "-o", type=str, required=True)
    parser.add_argument("--model-path", "-m", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--fps", "-f", type=int, default=VIDEO_TARGET_FPS)
    parser.add_argument("--video-width", "-W", type=int, default=VIDEO_RESIZE[1])
    parser.add_argument("--video-height", "-H", type=int, default=VIDEO_RESIZE[0])
    args = parser.parse_args()

    model = VideoPredictor()
    if not torch.cuda.is_available():
        print("CUDA is not available. Prediction will be performed on CPU, which may be slow.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    output_path = Path(args.output_video)
    temp_video_path = output_path.with_suffix(".silent.mp4")

    window_starts = _iter_audio_window_starts(args.input_audio, WINDOW_SECONDS)
    writer: Any = imageio.get_writer(str(temp_video_path), fps=args.fps)
    try:
        with torch.no_grad():
            for start_time in window_starts:
                audio_window = generate_spectrogram(
                    args.input_audio,
                    freq=AUDIO_FEATURES_PER_SECOND,
                    start_time=start_time,
                    duration=WINDOW_SECONDS,
                )
                audio_window = torch.from_numpy(audio_window).unsqueeze(0).unsqueeze(0).to(device)
                output = model(audio_window).squeeze(0).detach().cpu().numpy()

                frames = np.asarray(output)
                if frames.ndim == 4 and frames.shape[1] in (1, 3):
                    frames = np.transpose(frames, (0, 2, 3, 1))
                if frames.dtype != np.uint8:
                    frames = np.clip(frames, 0.0, 1.0)
                    frames = (frames * 255.0).astype(np.uint8)
                for frame in frames:
                    if frame.shape[0] != args.video_height or frame.shape[1] != args.video_width:
                        frame = np.asarray(Image.fromarray(frame).resize((args.video_width, args.video_height), Image.Resampling.BILINEAR))
                    writer.append_data(frame)
    finally:
        writer.close()

    _mux_audio_with_video(str(temp_video_path), args.input_audio, str(output_path))
    os.remove(temp_video_path)

if __name__ == "__main__":
    main()