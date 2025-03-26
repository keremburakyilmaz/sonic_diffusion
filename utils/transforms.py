import torchaudio
from torchvision import transforms as T

def get_image_transform(size = (224, 224)):
# image transformations for models
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def get_log_mel_transform(sample_rate=16000,
                          n_fft=1024,
                          win_length=None,
                          hop_length=512,
                          n_mels=64
                        ):
# audio transformations from raw to mel spectogram for models

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length or n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
