import os
import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np
import importlib.util

# 1) 按文件路径导入 Audio2Mel（因为目录名包含“-”不能用常规包导入）
module_path = "/root/code/encodec-pytorch/audio_to_mel.py"
spec = importlib.util.spec_from_file_location("audio_to_mel", module_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
Audio2Mel = m.Audio2Mel

# 2) 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3) 读取 WAV（保持原采样率；mono=False 保留多声道）
wav_path = "/root/autodl-tmp/test/output_dir/output_dir/GT_bw6.wav"
audio_np, sr = librosa.load(wav_path, sr=None, mono=False)  # shape: (T,) 或 (C, T)

# 4) 转成 torch.Tensor 并放到正确设备
audio_t = torch.from_numpy(audio_np).float().to(device)

# 5) 建立转换器（采样率用音频真实 sr）
mel_converter = Audio2Mel(sampling_rate=sr, device=device)

# 6) 计算 log-Mel
log_mel = mel_converter(audio_t)  # 可能是 [n_mel, T] 或 [C, n_mel, T]

# 7) 统一拿第一条通道绘图
if log_mel.ndim == 3:
    log_mel_for_plot = log_mel[0]          # [n_mel, T]
else:
    log_mel_for_plot = log_mel             # [n_mel, T]

log_mel_np = log_mel_for_plot.detach().cpu().numpy()

# 8) 可视化
plt.figure(figsize=(10, 4))
plt.imshow(log_mel_np, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='log10 Mel')
plt.title(f"Log-Mel Spectrogram ({os.path.basename(wav_path)})")
plt.xlabel("Frames")
plt.ylabel("Mel bins")
plt.tight_layout()
plt.show()

plt.savefig("/root/autodl-tmp/test/output_dir/output_dir/mel_vis.png")