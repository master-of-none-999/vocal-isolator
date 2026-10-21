"""PCM WAV helpers and FFmpeg decoding.

Demucs is fed a stereo 44.1 kHz 16-bit WAV that we decode ourselves, so the
separation step does not depend on Demucs shelling out to ffprobe.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from .errors import VocalError

TARGET_SAMPLE_RATE = 44_100
TARGET_CHANNELS = 2


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return float audio shaped (channels, samples) and the sample rate."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise VocalError("內部音訊必須是 16-bit PCM WAV。")
    if channels < 1 or not raw:
        raise VocalError("音訊是空的。")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    audio = samples.reshape(-1, channels).T.copy()
    return audio, sample_rate


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float audio shaped (channels, samples) as 16-bit PCM WAV."""
    if audio.ndim != 2:
        raise VocalError("無法寫入這段音訊。")
    clipped = np.clip(audio, -1.0, 1.0)
    interleaved = np.ascontiguousarray(clipped.T)
    pcm = np.rint(interleaved * 32767.0).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(audio.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def ffmpeg_path() -> Path:
    system = shutil.which("ffmpeg")
    if system:
        return Path(system)
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise VocalError("找不到 FFmpeg。請先執行 .\\setup.ps1。") from exc
    try:
        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except RuntimeError as exc:
        raise VocalError("找不到 FFmpeg。請先執行 .\\setup.ps1。") from exc


def extract_pcm_wav(source: Path, destination: Path) -> None:
    """Decode any supported media file to a stereo 44.1 kHz 16-bit WAV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg_path()),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    _run_command(command, "讀取音訊")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise VocalError("讀取音訊失敗，成品是空的。")


_AUDIO_CODECS = {
    "flac": ["-codec:a", "flac"],
    "mp3": ["-codec:a", "libmp3lame", "-b:a", "320k"],
    "m4a": ["-codec:a", "aac", "-b:a", "256k"],
}


def encode_audio(source_wav: Path, destination: Path, audio_format: str) -> None:
    """Encode a 16-bit WAV as FLAC, 320 kbps MP3, or 256 kbps AAC in an M4A."""
    codec = _AUDIO_CODECS.get(audio_format)
    if codec is None:
        raise VocalError("不支援這個輸出格式。可以使用：WAV、FLAC、MP3、M4A。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg_path()),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_wav),
        *codec,
        str(destination),
    ]
    _run_command(command, f"輸出 {audio_format.upper()}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise VocalError(f"輸出 {audio_format.upper()} 失敗，成品是空的。")


def encode_mp3(source_wav: Path, destination: Path) -> None:
    encode_audio(source_wav, destination, "mp3")


def _run_command(command: list[str], description: str) -> None:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )
    if process.returncode == 0:
        return
    combined = "\n".join(part for part in (process.stdout, process.stderr) if part)
    tail = "\n".join(combined.splitlines()[-20:])
    detail = f"\n\n{tail}" if tail else ""
    raise VocalError(f"{description}失敗。{detail}")
