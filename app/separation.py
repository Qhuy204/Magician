import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import settings
from .hardware import detect

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        from demucs.pretrained import get_model

        _model = get_model(settings.demucs_model)
        _model.eval()
    return _model


def separate(mix_wav: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Tách vocal và nhạc nền. Trả về (vocals, bgm, sr), mảng shape (n, 2)."""
    from demucs.apply import apply_model

    audio, sr = sf.read(mix_wav, dtype="float32", always_2d=True)
    with _lock:
        model = _get_model()
        if sr != model.samplerate:
            raise ValueError(f"expected {model.samplerate} Hz, got {sr}")
        wav = torch.from_numpy(audio.T.copy())
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            sources = apply_model(model, ((wav - mean) / std)[None], device=detect().demucs_device, split=True, overlap=0.25)[0]
        sources = (sources * std + mean).cpu().numpy()
        if detect().demucs_device.startswith("cuda"):
            torch.cuda.empty_cache()
    vocals = sources[model.sources.index("vocals")]
    bgm = sources.sum(0) - vocals
    return vocals.T.copy(), bgm.T.copy(), sr
