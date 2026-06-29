import torch
import torch.nn as nn
import torch.nn.functional as F

from video_prediction.constants import AUDIO_FEATURES_PER_VIDEO_FEATURE, FREQ_BINS, VIDEO_RESIZE, VIDEO_TARGET_FPS, WINDOW_SECONDS


class VideoPredictor(nn.Module):
    """A small RNN that maps an audio spectrogram window to a video clip.

    Input shape:
        (batch_size, 1, freq_bins, audio_time_steps)
    Output shape:
        (batch_size, video_frames, 3, height, width)
    """

    def __init__(self, hidden_size: int = 128, low_res_scale: int = 4):
        super().__init__()
        self.freq_bins = FREQ_BINS
        self.video_frames = int(WINDOW_SECONDS * VIDEO_TARGET_FPS)
        self.audio_chunk_size = int(AUDIO_FEATURES_PER_VIDEO_FEATURE)
        self.hidden_size = hidden_size

        self.low_res_height = VIDEO_RESIZE[0] // low_res_scale
        self.low_res_width = VIDEO_RESIZE[1] // low_res_scale
        self.low_res_channels = 3
        self.frame_vector_size = self.low_res_channels * self.low_res_height * self.low_res_width

        self.rnn = nn.RNN(
            input_size=self.freq_bins,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.frame_head = nn.Linear(self.hidden_size, self.frame_vector_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the audio sequence and decode one frame per audio chunk."""
        if x.dim() != 4:
            raise ValueError(f"Expected input shape (batch, channels, freq, time), got {tuple(x.shape)}")

        # Drop the channel axis and make the time axis explicit for the RNN.
        audio = x.squeeze(1)  # (B, F, T)
        audio = F.adaptive_avg_pool1d(audio, self.video_frames)  # (B, F, 32)
        audio = audio.transpose(1, 2).contiguous()  # (B, 32, F)

        rnn_out, _ = self.rnn(audio)  # (B, 32, H)
        frame_vectors = self.frame_head(rnn_out)  # (B, 32, C*H*W)
        frame_vectors = torch.sigmoid(frame_vectors)

        frames = frame_vectors.view(
            frame_vectors.size(0),
            self.video_frames,
            self.low_res_channels,
            self.low_res_height,
            self.low_res_width,
        )

        frames = frames.view(-1, self.low_res_channels, self.low_res_height, self.low_res_width)
        frames = F.interpolate(
            frames,
            size=VIDEO_RESIZE,
            mode="bilinear",
            align_corners=False,
        )
        frames = frames.view(-1, self.video_frames, self.low_res_channels, VIDEO_RESIZE[0], VIDEO_RESIZE[1])
        return frames
