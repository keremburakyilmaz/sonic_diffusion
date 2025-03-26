import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchaudio

class AudioImageDataset(Dataset):
    def __init__(self, root_dir, audio_transform = None, image_transform = None, sample_rate = 16000):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.audio_transform = audio_transform
        self.image_transform = image_transform

        # Match .wav files with corresponding .jpg or .png files
        self.pairs = []
        for fname in os.listdir(root_dir):
            if fname.endswith(".wav"):
                base = os.path.splitext(fname)[0]
                audio_path = os.path.join(root_dir, fname)

                for ext in [".jpg", ".png"]:
                    image_path = os.path.join(root_dir, base + ext)
                    if os.path.exists(image_path):
                        self.pairs.append((audio_path, image_path))
                        break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path, image_path = self.pairs[idx]

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if self.audio_transform:
            waveform = self.audio_transform(waveform)

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        return {
            "audio": waveform,        # Tensor shape: [1, T] or [2, T]
            "image": image,           # Tensor shape: [3, H, W]
            "audio_path": audio_path,
            "image_path": image_path,
        }