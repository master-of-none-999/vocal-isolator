"""Command-line entry: python -m vocal_isolator SONG --preset precise"""

from __future__ import annotations

import argparse
from pathlib import Path

from .errors import VocalError
from .pipeline import OUTPUT_DIR, separate_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在本機把人聲從歌曲分離出來")
    parser.add_argument("source", type=Path, help="本機音訊或影片")
    parser.add_argument(
        "--preset",
        choices=("precise", "standard"),
        default="precise",
        help="precise 使用 htdemucs_ft；standard 使用與 K 歌工房相同的 htdemucs",
    )
    parser.add_argument("--shifts", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--instrumental", action="store_true", help="同時輸出伴奏")
    parser.add_argument(
        "--format",
        choices=("wav", "flac", "mp3", "m4a"),
        default="wav",
        help="輸出格式。wav 與 flac 是 16-bit 無損，mp3 是 320 kbps，m4a 是 AAC 256 kbps",
    )
    parser.add_argument("--mp3", action="store_true", help="輸出 320 kbps MP3，等同 --format mp3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    def report(fraction: float, message: str) -> None:
        print(f"[{int(fraction * 100):3d}%] {message}", flush=True)

    try:
        result = separate_file(
            args.source,
            preset=args.preset,
            shifts=args.shifts,
            overlap=args.overlap,
            device=args.device,
            save_instrumental=args.instrumental,
            output_format=args.format,
            save_mp3=args.mp3,
            output_dir=args.output if args.output is not None else OUTPUT_DIR,
            progress=report,
            show_model_progress=True,
        )
    except VocalError as exc:
        print(f"錯誤：{exc}")
        return 1

    print(f"人聲：{result.vocals_path}")
    if result.instrumental_path:
        print(f"伴奏：{result.instrumental_path}")
    print(f"模型：{result.model_name}　裝置：{result.device}　長度：{result.duration:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
