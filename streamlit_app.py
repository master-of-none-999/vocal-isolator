"""Local-only vocal isolation window."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import streamlit as st

from vocal_isolator.errors import VocalError
from vocal_isolator.pipeline import BASE_DIR, MEDIA_SUFFIXES, normalize_media_path, separate_file

UPLOAD_TYPES = sorted(suffix.lstrip(".") for suffix in MEDIA_SUFFIXES)


def demucs_installed() -> bool:
    return importlib.util.find_spec("demucs") is not None and importlib.util.find_spec("torch") is not None


def accelerator_label() -> str:
    if importlib.util.find_spec("torch") is None:
        return "尚未安裝"
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def duration_label(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def resolve_source(uploaded, typed_path: str) -> Path | None:
    typed = normalize_media_path(typed_path) if typed_path.strip() else None
    if typed is not None and typed.is_file():
        return typed
    if uploaded is not None:
        destination = BASE_DIR / "work" / "uploads"
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / Path(uploaded.name).name
        target.write_bytes(uploaded.getvalue())
        return target
    return typed


st.set_page_config(page_title="人聲分離", page_icon="🎙️", layout="centered")
st.session_state.setdefault("vocal_result", None)

st.title("人聲分離")
st.caption("只在這台電腦上執行。歌曲不會上傳；Demucs 在本機把人聲從伴奏裡分開，並輸出人聲。")

ready = demucs_installed()
with st.sidebar:
    st.subheader("本機狀態")
    if ready:
        st.success("Demucs 已安裝")
    else:
        st.warning("Demucs 尚未安裝")
        st.caption("在這個資料夾執行 `.\\setup.ps1`，裝好後重新開啟。")
    st.write(f"運算裝置：{accelerator_label()}")
    st.caption("成品保存在 `outputs`。第一次分離會把模型下載到 `models`，之後可離線使用。")

with st.container(border=True):
    st.subheader("1. 選擇歌曲")
    uploaded = st.file_uploader(
        "音訊或影片",
        type=UPLOAD_TYPES,
        help="也可以在下面直接填本機路徑，較大的檔案會比較快。",
    )
    typed_path = st.text_input(
        "或填本機路徑",
        placeholder=r"D:\Music\song.mp3",
    )

with st.container(border=True):
    st.subheader("2. 分離方式")
    preset_label = st.radio(
        "模型",
        ["精確", "標準"],
        horizontal=True,
        help="精確使用 htdemucs_ft，是 htdemucs 的微調集成，人聲比較乾淨，也比較慢。標準使用與 K 歌工房相同的 htdemucs。",
    )
    format_label = st.radio(
        "輸出格式",
        ["WAV", "FLAC", "MP3", "M4A"],
        horizontal=True,
        help="WAV 與 FLAC 是 16-bit 無損。MP3 是 320 kbps。M4A 是 AAC 256 kbps。有勾選伴奏時，伴奏也用這個格式。",
    )
    with st.expander("進階設定"):
        shifts = st.select_slider("時間位移次數", options=[1, 2], value=1)
        overlap = st.select_slider("片段重疊", options=[0.25, 0.5], value=0.25)
        device_label = st.radio("運算裝置", ["自動", "GPU", "CPU"], horizontal=True)
        save_instrumental = st.checkbox("同時輸出伴奏")
    st.caption("兩種模型都用同一套推論：44.1 kHz 音訊、均值標準差正規化、一段位移、0.25 重疊、分段處理。差別是精確模式改留人聲，並用微調後的集成模型。")

rights_confirmed = st.checkbox("我有權處理這段音訊，並且只會依授權使用分離出來的人聲。")
start = st.button("分離人聲", type="primary", disabled=not rights_confirmed or not ready, width="stretch")

if start:
    st.session_state.vocal_result = None
    source = resolve_source(uploaded, typed_path)
    if source is None:
        st.error("請先選擇檔案，或填上本機路徑。")
    else:
        progress = st.progress(0, text="準備開始…")
        status = st.status("準備分離…", expanded=True)

        def update(fraction: float, message: str) -> None:
            progress.progress(int(fraction * 100), text=message)
            status.update(label=message, state="running")

        device = {"自動": "auto", "GPU": "cuda", "CPU": "cpu"}[device_label]
        try:
            result = separate_file(
                source,
                preset="precise" if preset_label == "精確" else "standard",
                shifts=int(shifts),
                overlap=float(overlap),
                device=device,
                save_instrumental=save_instrumental,
                output_format=format_label.lower(),
                progress=update,
            )
        except VocalError as exc:
            progress.empty()
            status.update(label="分離未完成", state="error", expanded=True)
            st.error(str(exc))
        except Exception as exc:
            progress.empty()
            status.update(label="發生未預期的錯誤", state="error", expanded=True)
            st.error("分離過程發生未預期的錯誤。")
            with st.expander("技術資料"):
                st.exception(exc)
        else:
            progress.progress(100, text="完成")
            status.update(label="人聲分離完成", state="complete", expanded=False)
            st.session_state.vocal_result = result

AUDIO_MIME = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
}

result = st.session_state.vocal_result
if result and Path(result.vocals_path).exists():
    mime = AUDIO_MIME.get(result.output_format, "audio/wav")
    format_name = result.output_format.upper()
    st.subheader("人聲")
    st.audio(Path(result.vocals_path).read_bytes(), format=mime)
    columns = st.columns(3)
    columns[0].metric("長度", duration_label(result.duration))
    columns[1].metric("模型", result.model_name)
    columns[2].metric("裝置", "GPU" if result.device == "cuda" else "CPU")
    if result.peak < 0.01:
        st.warning("人聲非常小。這首歌的主唱可能很弱，或模型沒有把人聲分出來。可以改用另一個模型再試一次。")
    st.download_button(
        f"下載人聲 {format_name}",
        data=Path(result.vocals_path).read_bytes(),
        file_name=Path(result.vocals_path).name,
        mime=mime,
        type="primary",
        width="stretch",
    )
    if result.instrumental_path and Path(result.instrumental_path).exists():
        st.subheader("伴奏")
        st.audio(Path(result.instrumental_path).read_bytes(), format=mime)
        st.download_button(
            f"下載伴奏 {format_name}",
            data=Path(result.instrumental_path).read_bytes(),
            file_name=Path(result.instrumental_path).name,
            mime=mime,
            width="stretch",
        )

st.caption("分離品質取決於原曲混音。和聲、很重的混響，以及和樂器疊在一起的人聲，仍可能留下一點殘響。")
