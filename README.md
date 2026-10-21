# 人聲分離

只在這台電腦上運行的人聲分離工具。把本機歌曲或影片裡的人聲抽出來，可存成 WAV、FLAC、MP3 或 M4A。

分離方式以 Demucs 流程：先把媒體解成 44.1 kHz、雙聲道、16-bit PCM，再用 Hybrid Transformer Demucs（`htdemucs`）做分段推論，設定是 1 次時間位移、0.25 重疊。K 歌工房把人聲以外的音軌加總，留下伴奏；這個 App 留下人聲。

精確模式改用同一個模型的微調集成版 `htdemucs_ft`。它比較慢，人聲通常更乾淨。

歌曲不會上傳。程式只聽 `127.0.0.1`。第一次分離需要連線，把模型下載到 `models`；之後可以離線使用。

請只處理你有權使用的音訊。

## 開始使用

這台電腦若有 NVIDIA RTX 50 系列顯卡，安裝腳本會裝 CUDA 12.8 版 PyTorch，分離才會用到 GPU。

```powershell
.\setup.ps1
.\start.ps1
```

也可以雙擊 `start.bat`。瀏覽器會開啟本機頁面。

沒有 NVIDIA GPU 時：

```powershell
.\setup.ps1 -Device cpu
```

命令列：

```powershell
.\.venv\Scripts\python.exe -m vocal_isolator "D:\Music\song.mp3" --preset precise
```

`--preset standard` 改用與 K 歌工房相同的 `htdemucs`。加上 `--instrumental` 會同時輸出伴奏。`--format wav|flac|mp3|m4a` 選擇成品格式，預設是 WAV；`--mp3` 等同 `--format mp3`。

## 輸出

人聲放在 `outputs`。WAV 與 FLAC 是 16-bit 無損，MP3 是 320 kbps，M4A 是 AAC 256 kbps。處理中的暫存檔放在 `work\jobs`，成功後會刪除；失敗時會留下，方便查看。

## 測試

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

## 限制

和聲、大量混響，以及和樂器疊得很緊的人聲，模型仍可能留下殘響。這是來源分離的上限，不是漏做了後製步驟。
