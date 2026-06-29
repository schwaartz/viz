import argparse
import torch
import numpy as np
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
    pass


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
    parser.add_argument("--video-width", "-w", type=int, default=VIDEO_RESIZE[0])
    parser.add_argument("--video-height", "-h", type=int, default=VIDEO_RESIZE[1])
    args = parser.parse_args()

    model = VideoPredictor()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    if not torch.cuda.is_available():
        print("CUDA is not available. Prediction will be performed on CPU, which may be slow.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input = generate_spectrogram(
        args.input_audio,
        freq=AUDIO_FEATURES_PER_SECOND,
        start_time=0,
        duration=None)
    input = torch.from_numpy(input)
    input = input.unsqueeze(0)  # Add batch dimension
    
    output = model(input.to(device))
    output.squeeze(0)  # Remove batch dimension
    output.detach().cpu().numpy()  # Convert to numpy array

    save_video(output, args.output_video, fps=args.fps, width=args.video_width, height=args.video_height)

if __name__ == "__main__":
    main()