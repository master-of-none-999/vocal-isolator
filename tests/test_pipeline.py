from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from vocal_isolator.audio_io import load_wav, save_wav
from vocal_isolator.errors import VocalError
from vocal_isolator.pipeline import separate_file


def test_separate_file_writes_vocals_and_removes_the_job(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "demo track.wav"
    source.write_bytes(b"not-a-real-wav")
    output_dir = tmp_path / "outputs"

    def fake_extract(media: Path, destination: Path) -> None:
        assert media == source
        tone = np.zeros((2, 44_100), dtype=np.float32)
        tone[0, :1000] = 0.5
        save_wav(destination, tone, 44_100)

    def fake_separate(mixture, vocals_path, instrumental_path, **kwargs) -> str:
        audio, sample_rate = load_wav(mixture)
        assert kwargs["model_name"] == "htdemucs_ft"
        assert kwargs["shifts"] == 1
        assert kwargs["overlap"] == 0.25
        save_wav(vocals_path, audio * 0.5, sample_rate)
        assert instrumental_path is None
        return "cuda"

    monkeypatch.setattr("vocal_isolator.pipeline.extract_pcm_wav", fake_extract)
    monkeypatch.setattr("vocal_isolator.pipeline.separate_mixture", fake_separate)
    monkeypatch.setattr("vocal_isolator.pipeline.WORK_DIR", tmp_path / "jobs")

    messages: list[str] = []
    result = separate_file(
        source,
        preset="precise",
        output_dir=output_dir,
        progress=lambda _fraction, message: messages.append(message),
    )

    assert result.vocals_path.is_file()
    assert result.vocals_path.parent == output_dir
    assert "人聲" in result.vocals_path.name
    assert result.device == "cuda"
    assert result.model_name == "htdemucs_ft"
    assert result.duration == pytest.approx(1.0)
    assert result.peak == pytest.approx(0.25)
    assert messages[0].startswith("正在把歌曲")
    assert messages[-1] == "人聲分離完成。"
    assert list((tmp_path / "jobs").glob("*")) == []


def test_unknown_suffix_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("hello", encoding="utf-8")
    with pytest.raises(VocalError, match="不支援這個格式"):
        separate_file(source, output_dir=tmp_path / "out")


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp3"
    with pytest.raises(VocalError, match=re.escape(str(missing))):
        separate_file(f'"{missing}"', output_dir=tmp_path / "out")


def test_flac_output_uses_the_encoder(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "song.wav"
    source.write_bytes(b"wrapper")
    encoded: list[str] = []

    def fake_extract(media: Path, destination: Path) -> None:
        save_wav(destination, np.zeros((2, 1000), dtype=np.float32), 44_100)

    def fake_separate(mixture, vocals_path, instrumental_path, **kwargs) -> str:
        audio, sample_rate = load_wav(mixture)
        save_wav(vocals_path, audio, sample_rate)
        save_wav(instrumental_path, audio, sample_rate)
        return "cpu"

    def fake_encode(source_wav: Path, destination: Path, audio_format: str) -> None:
        encoded.append(audio_format)
        destination.write_bytes(b"encoded")

    monkeypatch.setattr("vocal_isolator.pipeline.extract_pcm_wav", fake_extract)
    monkeypatch.setattr("vocal_isolator.pipeline.separate_mixture", fake_separate)
    monkeypatch.setattr("vocal_isolator.pipeline.encode_audio", fake_encode)
    monkeypatch.setattr("vocal_isolator.pipeline.WORK_DIR", tmp_path / "jobs")

    result = separate_file(
        source,
        output_format="flac",
        save_instrumental=True,
        output_dir=tmp_path / "out",
    )

    assert result.output_format == "flac"
    assert result.vocals_path.suffix == ".flac"
    assert result.instrumental_path is not None
    assert result.instrumental_path.suffix == ".flac"
    assert encoded == ["flac", "flac"]
    assert list((tmp_path / "out").glob("*.wav")) == []


def test_mp3_flag_selects_mp3(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "song.wav"
    source.write_bytes(b"wrapper")

    def fake_extract(media: Path, destination: Path) -> None:
        save_wav(destination, np.zeros((2, 1000), dtype=np.float32), 44_100)

    def fake_separate(mixture, vocals_path, instrumental_path, **kwargs) -> str:
        audio, sample_rate = load_wav(mixture)
        save_wav(vocals_path, audio, sample_rate)
        return "cpu"

    monkeypatch.setattr("vocal_isolator.pipeline.extract_pcm_wav", fake_extract)
    monkeypatch.setattr("vocal_isolator.pipeline.separate_mixture", fake_separate)
    monkeypatch.setattr(
        "vocal_isolator.pipeline.encode_audio",
        lambda source_wav, destination, audio_format: destination.write_bytes(b"mp3"),
    )
    monkeypatch.setattr("vocal_isolator.pipeline.WORK_DIR", tmp_path / "jobs")

    result = separate_file(source, save_mp3=True, output_dir=tmp_path / "out")
    assert result.output_format == "mp3"
    assert result.vocals_path.suffix == ".mp3"


def test_unknown_output_format_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "song.wav"
    source.write_bytes(b"wrapper")
    with pytest.raises(VocalError, match="不支援這個輸出格式"):
        separate_file(source, output_format="ogg", output_dir=tmp_path / "out")


def test_safe_filename_strips_windows_characters() -> None:
    from vocal_isolator.pipeline import safe_filename

    assert safe_filename('a<>:"/\\|?*b') == "a b"
    assert safe_filename("   ") == "vocals"
    assert safe_filename("CON") == "_CON"
