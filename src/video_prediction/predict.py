import argparse
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

    audio_features = generate_spectrogram(
        args.input_audio,
        freq=AUDIO_FEATURES_PER_SECOND,
        start_time=0,
        duration=None,
    )
    audio_features = torch.from_numpy(audio_features).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(audio_features)
        output = output.squeeze(0).detach().cpu().numpy()

    save_video(output, args.output_video, fps=args.fps, width=args.video_width, height=args.video_height)

if __name__ == "__main__":
    main()