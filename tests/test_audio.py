from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from vocal_isolator.audio_io import load_wav, save_wav
from vocal_isolator.separator import isolate_vocals


def test_wav_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    source = np.array([[0.0, 0.25, -0.25], [0.5, -0.5, 0.0]], dtype=np.float32)

    save_wav(path, source, 44_100)
    restored, sample_rate = load_wav(path)

    assert sample_rate == 44_100
    assert restored.shape == source.shape
    assert np.allclose(restored, source, atol=1 / 32768)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 2


def test_isolate_vocals_keeps_the_vocal_stem() -> None:
    sources = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 2.0]],
            [[0.25, 0.5], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    vocals, instrumental = isolate_vocals(sources, ["drums", "vocals", "other"])

    assert np.allclose(vocals, [[0.0, 0.0], [2.0, 2.0]])
    assert np.allclose(instrumental, [[1.25, 1.5], [0.0, 0.0]])
