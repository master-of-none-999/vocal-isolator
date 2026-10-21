"""Demucs vocal isolation.

The inference settings match the local runner in karaoke-factory: load a known
PCM WAV, normalize by the mixture mean and standard deviation, then call
``apply_model`` with one shift, a 0.25 overlap and chunked processing. That
project sums every non-vocal stem and keeps the instrumental. This app keeps
the vocal stem instead.

The precise preset uses ``htdemucs_ft``, the fine-tuned ensemble of the same
Hybrid Transformer model (``htdemucs``) that karaoke-factory runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .audio_io import load_wav, save_wav
from .errors import VocalError

MODEL_NAMES = ("htdemucs", "htdemucs_ft")
_CACHED: dict[str, object] = {"name": "", "model": None}


def isolate_vocals(
    sources: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the vocal stem and sum every other stem into an instrumental."""
    if sources.ndim != 3:
        raise VocalError("分離結果的形狀不正確。")
    if len(names) != sources.shape[0]:
        raise VocalError("音軌名稱與分離結果數量不符。")
    vocal_index = next((index for index, name in enumerate(names) if name == "vocals"), None)
    if vocal_index is None:
        raise VocalError("這個模型沒有人聲軌道。")
    vocals = sources[vocal_index]
    others = [sources[index] for index, name in enumerate(names) if name != "vocals"]
    if others:
        instrumental = np.sum(np.stack(others, axis=0), axis=0).astype(np.float32, copy=False)
    else:
        instrumental = np.zeros_like(vocals)
    return vocals, instrumental


def configure_model_cache(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(models_dir)


def separate_mixture(
    mixture_path: Path,
    vocals_path: Path,
    instrumental_path: Path | None,
    *,
    model_name: str,
    shifts: int,
    overlap: float,
    device: str,
    models_dir: Path,
    show_progress: bool = False,
) -> str:
    """Separate one WAV and return the device name that actually ran."""
    if model_name not in MODEL_NAMES:
        raise VocalError("不支援這個分離模型。")
    if shifts < 1 or shifts > 5:
        raise VocalError("位移次數必須介於 1 到 5。")
    if not 0 <= overlap < 1:
        raise VocalError("重疊比例必須介於 0 到 1 之間。")

    configure_model_cache(models_dir)
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise VocalError(
            "尚未安裝 Demucs。請關閉程式後執行 .\\setup.ps1。"
        ) from exc

    resolved = _resolve_device(device, torch)
    model = _load_model(model_name, get_model)
    mixture, sample_rate = load_wav(mixture_path)
    if mixture.shape[1] < sample_rate // 10:
        raise VocalError("音訊太短，無法分離。請提供至少約 0.1 秒的內容。")

    audio = torch.from_numpy(mixture)
    audio = convert_audio(audio, sample_rate, model.samplerate, model.audio_channels)
    reference = audio.mean(0)
    mean = reference.mean()
    std = reference.std().clamp_min(1e-8)
    normalised = (audio - mean) / std

    try:
        with torch.inference_mode():
            estimated = apply_model(
                model,
                normalised[None],
                device=resolved,
                shifts=shifts,
                split=True,
                overlap=overlap,
                progress=show_progress,
                num_workers=0,
            )[0]
    except RuntimeError as exc:
        raise VocalError(f"人聲分離失敗：{exc}") from exc

    estimated = estimated * std + mean
    names = list(model.sources)
    vocals, instrumental = isolate_vocals(estimated.detach().cpu().numpy(), names)
    save_wav(vocals_path, vocals, int(model.samplerate))
    if instrumental_path is not None:
        save_wav(instrumental_path, instrumental, int(model.samplerate))
    return resolved


def _resolve_device(preference: str, torch_module: object) -> str:
    cuda_ready = bool(torch_module.cuda.is_available())
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not cuda_ready:
            raise VocalError("找不到可用的 NVIDIA GPU。請改為自動或 CPU。")
        return "cuda"
    if preference != "auto":
        raise VocalError("運算裝置必須是自動、GPU 或 CPU。")
    return "cuda" if cuda_ready else "cpu"


def _load_model(model_name: str, get_model):
    if _CACHED["name"] != model_name or _CACHED["model"] is None:
        try:
            model = get_model(model_name)
        except Exception as exc:
            raise VocalError(
                "無法載入 Demucs 模型。第一次使用需要連線，模型會下載到本機的 models 資料夾；"
                f"下載完成後就可以離線分離。\n\n{exc}"
            ) from exc
        model.eval()
        model.cpu()
        _CACHED["name"] = model_name
        _CACHED["model"] = model
    return _CACHED["model"]
