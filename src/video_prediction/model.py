import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoPredictor(nn.Module):
    """
    Takes in a spectrogram and predicts the next frame in the sequence based on its previous frames.
    """
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(VideoPredictor, self).__init__()
        pass

    def forward(self, x):
        pass