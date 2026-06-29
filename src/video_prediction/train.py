import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
import argparse
from pathlib import Path
from video_prediction.model import VideoPredictor
from video_prediction.dataset import CachedClipDataset
from video_prediction.constants import (
    DEFAULT_LR,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_PATH,
)

def train(model: Module, dataset: Dataset, epochs: int, batch_size: int, lr: float):
    """
    Starts a training loop for the video prediction model using the specified
    model and dataset.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Training will be performed on CPU, which may be slow.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    model.to(device)
    model.train()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            audio = batch["audio"].to(device, non_blocking=use_amp)
            video = batch["video"].to(device, non_blocking=use_amp)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(audio)
                loss = loss_func(output, video)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 100 == 0:
                print(f"EPOCH {epoch + 1}/{epochs}, BATCH {i}/{len(dataloader)}, LOSS: {loss.item()}")


def main():
    """
    Main function to initialize the model, dataset, and start the training process.
    """
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", "-l", type=float, default=DEFAULT_LR)
    parser.add_argument("--manifest-path", type=str, default="video_prediction/data/manifest.jsonl")
    args = parser.parse_args()

    model = VideoPredictor()
    dataset = CachedClipDataset(manifest_path=args.manifest_path)
    print(f"Starting training loop:",
          f"\n\t- Epochs: {args.epochs}",
          f"\n\t- Batches: {len(dataset)//args.batch_size}",
          f"\n\t- Learning Rate: {args.lr}")
    train(model, dataset, args.epochs, args.batch_size, args.lr)

    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), str(save_path))

if __name__ == "__main__":
    main()