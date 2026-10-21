"""Turn one local media file into a vocal WAV."""

from __future__ import annotations

import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import TARGET_SAMPLE_RATE, encode_audio, extract_pcm_wav, load_wav
from .errors import VocalError
from .separator import separate_mixture

BASE_DIR = Path(__file__).resolve().parent.parent
WORK_DIR = BASE_DIR / "work" / "jobs"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

MEDIA_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}

OUTPUT_FORMATS = ("wav", "flac", "mp3", "m4a")

PRESETS = {
    "precise": {
        "model_name": "htdemucs_ft",
        "shifts": 1,
        "overlap": 0.25,
    },
    "standard": {
        "model_name": "htdemucs",
        "shifts": 1,
        "overlap": 0.25,
    },
}

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class SeparationResult:
    vocals_path: Path
    instrumental_path: Path | None
    output_format: str
    source_name: str
    duration: float
    sample_rate: int
    model_name: str
    device: str
    peak: float


def normalize_media_path(value: str | Path) -> Path:
    """Accept paths copied from Explorer, a browser, or PowerShell."""
    text = str(value).strip().lstrip("\ufeff")
    if text.lower().startswith("file:"):
        from urllib.parse import unquote, urlparse

        parsed = urlparse(text)
        text = unquote(parsed.path)
        if len(text) >= 3 and text[0] == "/" and text[2] == ":":
            text = text[1:]
    if text.startswith("& "):
        text = text[2:]
    text = text.strip().strip("\"'「」『』").strip()
    return Path(text).expanduser()


def separate_file(
    source: Path,
    *,
    preset: str = "precise",
    model_name: str | None = None,
    shifts: int | None = None,
    overlap: float | None = None,
    device: str = "auto",
    save_instrumental: bool = False,
    output_format: str = "wav",
    save_mp3: bool = False,
    output_dir: Path = OUTPUT_DIR,
    progress: ProgressCallback | None = None,
    show_model_progress: bool = False,
) -> SeparationResult:
    """Separate ``source`` and write the vocal stem under ``output_dir``."""
    source = normalize_media_path(source)
    if preset not in PRESETS:
        raise VocalError("不支援這個分離預設。")
    if not source.is_file():
        raise VocalError(f"找不到這個檔案：{source}")
    if source.suffix.lower() not in MEDIA_SUFFIXES:
        allowed = "、".join(sorted(suffix.lstrip(".") for suffix in MEDIA_SUFFIXES))
        raise VocalError(f"不支援這個格式。可以使用：{allowed}。")

    resolved_format = "mp3" if save_mp3 and output_format == "wav" else output_format.lower()
    if resolved_format not in OUTPUT_FORMATS:
        raise VocalError("不支援這個輸出格式。可以使用：WAV、FLAC、MP3、M4A。")

    chosen = PRESETS[preset]
    resolved_model = model_name or chosen["model_name"]
    resolved_shifts = chosen["shifts"] if shifts is None else shifts
    resolved_overlap = chosen["overlap"] if overlap is None else overlap

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir = WORK_DIR / uuid.uuid4().hex
    job_dir.mkdir(parents=True)
    succeeded = False

    try:
        _report(progress, 0.08, "正在把歌曲轉成 44.1 kHz 音訊…")
        mixture = job_dir / "mixture.wav"
        extract_pcm_wav(source, mixture)

        _report(progress, 0.22, "正在載入 Demucs 模型。第一次會下載到本機…")
        temp_vocals = job_dir / "vocals.wav"
        temp_instrumental = job_dir / "instrumental.wav" if save_instrumental else None
        used_device = separate_mixture(
            mixture,
            temp_vocals,
            temp_instrumental,
            model_name=resolved_model,
            shifts=resolved_shifts,
            overlap=resolved_overlap,
            device=device,
            models_dir=MODELS_DIR,
            show_progress=show_model_progress,
        )

        _report(progress, 0.9, "正在寫出人聲檔…")
        vocals, sample_rate = load_wav(temp_vocals)
        peak = float(np.max(np.abs(vocals))) if vocals.size else 0.0
        duration = vocals.shape[1] / sample_rate if sample_rate else 0.0
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = safe_filename(source.stem, "vocals")
        suffix = f".{resolved_format}"
        vocals_path = output_dir / f"{stem} 人聲 {stamp}{suffix}"
        _publish_audio(temp_vocals, vocals_path, resolved_format)

        instrumental_path = None
        if temp_instrumental is not None and temp_instrumental.is_file():
            instrumental_path = output_dir / f"{stem} 伴奏 {stamp}{suffix}"
            _publish_audio(temp_instrumental, instrumental_path, resolved_format)

        _report(progress, 1.0, "人聲分離完成。")
        succeeded = True
        return SeparationResult(
            vocals_path=vocals_path,
            instrumental_path=instrumental_path,
            output_format=resolved_format,
            source_name=source.name,
            duration=duration,
            sample_rate=sample_rate or TARGET_SAMPLE_RATE,
            model_name=resolved_model,
            device=used_device,
            peak=peak,
        )
    finally:
        if succeeded:
            shutil.rmtree(job_dir, ignore_errors=True)


def _publish_audio(source_wav: Path, destination: Path, audio_format: str) -> None:
    if audio_format == "wav":
        shutil.move(str(source_wav), destination)
        return
    encode_audio(source_wav, destination, audio_format)


def safe_filename(value: str, fallback: str = "vocals") -> str:
    cleaned = unicodedata.normalize("NFKC", value)
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    if cleaned.upper() in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}:
        cleaned = f"_{cleaned}"
    return cleaned[:80].rstrip(" .")


def _report(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback:
        callback(max(0.0, min(1.0, fraction)), message)
