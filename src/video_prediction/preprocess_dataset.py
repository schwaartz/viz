import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from video_prediction.audio_preprocessing import generate_spectrogram
from video_prediction.video_preprocessing import read_video_frames

WINDOW_SECONDS = 4.0
AUDIO_FEATURES_PER_SECOND = 32.0
VIDEO_TARGET_FPS = 8.0
VIDEO_RESIZE = (128, 128)


def _pair_media_files(audio_dir: Path, video_dir: Path) -> List[Tuple[Path, Path]]:
    audio_map = {path.stem: path for path in audio_dir.glob("*.mp3")}
    video_map = {path.stem: path for path in video_dir.glob("*.mp4")}
    common_stems = sorted(audio_map.keys() & video_map.keys())
    return [(audio_map[stem], video_map[stem]) for stem in common_stems]


def _window_starts(duration: float, window_seconds: float, stride_seconds: float) -> Iterable[float]:
    last_start = duration - window_seconds
    if last_start < 0:
        return []

    starts = []
    start = 0.0
    while start <= last_start + 1e-9:
        starts.append(round(start, 6))
        start += stride_seconds
    return starts


def build_dataset(
    audio_dir: str,
    video_dir: str,
    output_dir: str,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = WINDOW_SECONDS,
    audio_features_per_second: float = AUDIO_FEATURES_PER_SECOND,
    video_target_fps: float = VIDEO_TARGET_FPS,
    video_resize: Tuple[int, int] = VIDEO_RESIZE,
) -> Path:
    """Preprocess paired audio/video files into cached 4-second training samples."""
    audio_root = Path(audio_dir)
    video_root = Path(video_dir)
    output_root = Path(output_dir)
    samples_root = output_root / "samples"
    output_root.mkdir(parents=True, exist_ok=True)
    samples_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.jsonl"
    pairs = _pair_media_files(audio_root, video_root)

    written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for audio_path, video_path in pairs:
            # Skip clips that cannot provide a full 4-second window.
            import librosa

            duration = float(librosa.get_duration(path=str(audio_path)))
            if duration < window_seconds:
                continue

            for start_time in _window_starts(duration, window_seconds, stride_seconds):
                audio = generate_spectrogram(
                    audio_file=str(audio_path),
                    freq=audio_features_per_second,
                    start_time=start_time,
                    duration=window_seconds,
                )
                video = read_video_frames(
                    video_path=str(video_path),
                    target_fps=video_target_fps,
                    resize=video_resize,
                    start_time=start_time,
                    duration=window_seconds,
                )

                sample_name = f"{audio_path.stem}_{written:06d}.npz"
                sample_path = samples_root / sample_name
                np.savez_compressed(
                    sample_path,
                    audio=audio,
                    video=video,
                    source_audio=str(audio_path),
                    source_video=str(video_path),
                    start_time=start_time,
                    duration=window_seconds,
                )

                manifest.write(
                    json.dumps(
                        {
                            "sample_path": str(sample_path),
                            "source_audio": str(audio_path),
                            "source_video": str(video_path),
                            "start_time": start_time,
                            "duration": window_seconds,
                        }
                    )
                    + "\n"
                )
                written += 1

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached 4-second audio/video clips.")
    parser.add_argument("--audio-dir", default="input")
    parser.add_argument("--video-dir", default="output")
    parser.add_argument("--output-dir", default="video_prediction/data")
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--audio-features-per-second", type=float, default=AUDIO_FEATURES_PER_SECOND)
    parser.add_argument("--video-target-fps", type=float, default=VIDEO_TARGET_FPS)
    parser.add_argument("--video-width", type=int, default=VIDEO_RESIZE[1])
    parser.add_argument("--video-height", type=int, default=VIDEO_RESIZE[0])
    args = parser.parse_args()

    manifest_path = build_dataset(
        audio_dir=args.audio_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        audio_features_per_second=args.audio_features_per_second,
        video_target_fps=args.video_target_fps,
        video_resize=(args.video_height, args.video_width),
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
