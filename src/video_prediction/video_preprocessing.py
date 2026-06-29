import os
from typing import Optional, Tuple

import imageio
import numpy as np
from PIL import Image


def read_video_frames(
    video_path: str,
    target_fps: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    to_grayscale: bool = False,
    start_time: float = 0.0,
    duration: Optional[float] = None,
) -> np.ndarray:
    """
    Read a fixed time window from a video and return it as a tensor.

    Returns a NumPy array with shape (T, C, H, W) and float32 values in [0, 1].
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    reader = imageio.get_reader(video_path, "ffmpeg")
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 0) or 0)

    if src_fps <= 0:
        reader.close()
        raise ValueError(f"Could not determine fps for video: {video_path}")

    start_frame = int(round(start_time * src_fps))
    if duration is not None and target_fps is not None and max_frames is None:
        max_frames = int(round(duration * target_fps))

    if max_frames is None:
        if duration is not None and src_fps > 0:
            max_frames = int(round(duration * src_fps))
        else:
            try:
                max_frames = int(reader.count_frames()) - start_frame
            except Exception:
                max_frames = 0

    if target_fps and target_fps > 0:
        frame_timestamps = [start_time + (i / float(target_fps)) for i in range(max_frames)]
    else:
        frame_timestamps = [start_time + (i / src_fps) for i in range(max_frames)]

    frames = []
    for timestamp in frame_timestamps:
        index = int(round(timestamp * src_fps))
        try:
            frame = reader.get_data(index)
        except IndexError:
            break

        img = Image.fromarray(frame)
        img = img.convert("L" if to_grayscale else "RGB")

        if resize:
            # resize is (height, width)
            img = img.resize((resize[1], resize[0]), Image.BILINEAR)

        arr = np.asarray(img).astype(np.float32) / 255.0
        if to_grayscale:
            arr = arr[:, :, None]

        frames.append(np.transpose(arr, (2, 0, 1)))

        if max_frames and len(frames) >= max_frames:
            break

    reader.close()

    if len(frames) == 0:
        channels = 1 if to_grayscale else 3
        return np.zeros((0, channels, 0, 0), dtype=np.float32)

    return np.stack(frames, axis=0)


def save_sequence(path: str, seq: np.ndarray) -> None:
    """Save a sequence tensor to a compressed .npz file."""
    np.savez_compressed(path, frames=seq)


def load_sequence(path: str) -> np.ndarray:
    """Load a sequence saved with `save_sequence` and return the frames array."""
    with np.load(path) as d:
        return d["frames"]


def video_to_dataset_item(
    video_path: str,
    target_fps: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    to_grayscale: bool = False,
    start_time: float = 0.0,
    duration: Optional[float] = None,
) -> np.ndarray:
    """Convenience wrapper that returns a training-ready video tensor."""
    return read_video_frames(
        video_path=video_path,
        target_fps=target_fps,
        resize=resize,
        max_frames=max_frames,
        to_grayscale=to_grayscale,
        start_time=start_time,
        duration=duration,
    )
