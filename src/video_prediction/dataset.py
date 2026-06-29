import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset

from video_prediction.constants import (
    AUDIO_FEATURES_PER_SECOND,
    FREQ_BINS,
    VIDEO_RESIZE,
    VIDEO_TARGET_FPS,
    WINDOW_SECONDS,
)


@dataclass(frozen=True)
class ClipRecord:
    """
    Immutable class representing a single audio/video clip record in the dataset manifest.
    """
    sample_path: str
    source_audio: str
    source_video: str
    start_time: float
    duration: float


def load_manifest(manifest_path: str) -> List[ClipRecord]:
    """
    Loads a manifest file containing audio/video clip records and returns a list of ClipRecord instances.
    """
    records: List[ClipRecord] = []
    manifest_root = Path(manifest_path).resolve().parent
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            sample_path = Path(item["sample_path"])
            if not sample_path.is_absolute():
                if sample_path.exists():
                    sample_path = sample_path.resolve()
                else:
                    sample_path = (manifest_root / sample_path).resolve()
            source_audio = Path(item["source_audio"])
            if not source_audio.is_absolute():
                if source_audio.exists():
                    source_audio = source_audio.resolve()
                else:
                    source_audio = (manifest_root / source_audio).resolve()
            source_video = Path(item["source_video"])
            if not source_video.is_absolute():
                if source_video.exists():
                    source_video = source_video.resolve()
                else:
                    source_video = (manifest_root / source_video).resolve()
            records.append(
                ClipRecord(
                    sample_path=str(sample_path),
                    source_audio=str(source_audio),
                    source_video=str(source_video),
                    start_time=float(item["start_time"]),
                    duration=float(item["duration"]),
                )
            )
    return records


class CachedClipDataset(Dataset):
    """
    Load cached 4-second audio/video windows from a manifest.
    """
    def __init__(self, manifest_path: str):
        if torch is None:
            raise ImportError("torch is required to use CachedClipDataset")
        self.manifest_path = manifest_path
        self.records = self._filter_valid_records(load_manifest(manifest_path))

    def _filter_valid_records(self, records: List[ClipRecord]) -> List[ClipRecord]:
        valid_records: List[ClipRecord] = []
        expected_audio_shape = (FREQ_BINS, int(WINDOW_SECONDS * AUDIO_FEATURES_PER_SECOND))
        expected_video_shape = (int(WINDOW_SECONDS * VIDEO_TARGET_FPS), 3, VIDEO_RESIZE[0], VIDEO_RESIZE[1])

        for record in records:
            try:
                with np.load(record.sample_path) as data:
                    audio = data["audio"]
                    video = data["video"]
                if audio.shape == expected_audio_shape and video.shape == expected_video_shape:
                    valid_records.append(record)
            except Exception:
                continue

        return valid_records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        with np.load(record.sample_path) as data:
            # Add a channel dimension so the spectrogram is ready for CNN-style models.
            audio = torch.from_numpy(data["audio"]).float().unsqueeze(0)
            video = torch.from_numpy(data["video"]).float()

        return {
            "audio": audio,
            "video": video,
            "source_audio": record.source_audio,
            "source_video": record.source_video,
            "start_time": record.start_time,
            "duration": record.duration,
        }
