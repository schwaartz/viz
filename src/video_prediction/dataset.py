import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - lets preprocessing run without torch
    torch = None
    Dataset = object


@dataclass(frozen=True)
class ClipRecord:
    sample_path: str
    source_audio: str
    source_video: str
    start_time: float
    duration: float


def load_manifest(manifest_path: str) -> List[ClipRecord]:
    records: List[ClipRecord] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                ClipRecord(
                    sample_path=item["sample_path"],
                    source_audio=item["source_audio"],
                    source_video=item["source_video"],
                    start_time=float(item["start_time"]),
                    duration=float(item["duration"]),
                )
            )
    return records


class CachedClipDataset(Dataset):
    """Load cached 4-second audio/video windows from a manifest."""

    def __init__(self, manifest_path: str):
        if torch is None:
            raise ImportError("torch is required to use CachedClipDataset")
        self.manifest_path = manifest_path
        self.records = load_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        with np.load(record.sample_path) as data:
            audio = torch.from_numpy(data["audio"]).float()
            video = torch.from_numpy(data["video"]).float()

        return {
            "audio": audio,
            "video": video,
            "source_audio": record.source_audio,
            "source_video": record.source_video,
            "start_time": record.start_time,
            "duration": record.duration,
        }
