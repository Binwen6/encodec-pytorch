import torch
import torch.nn as nn

# 若本地已安装 mamba-ssm，则可正常导入；否则退化为恒等模块
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    class Mamba(nn.Module):
        """Fallback Mamba: 如果未安装 mamba_ssm，则该模块不改变输入。"""
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
        def forward(self, x):
            return x

class EEGEncoder(nn.Module):
    """
    使用 CNN + Mamba 架构的 EEG 编码器，将 8 秒钟、32 通道、500 Hz 采样率
    的 EEG 序列映射到与 EnCodec latent 空间形状一致的 [batch, 128, 75] 表示。
    """
    def __init__(self, channels: int = 32,
                 d_model: int = 256,
                 latent_dim: int = 128,
                 target_frames: int = 75):
        super().__init__()
        # 卷积下采样前端：多层卷积减少时间步长
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(1, 64),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, 128),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
        )
        # 最后一层卷积用 stride=4，总下采样比例 2×2×2×4=32，
        # 4000 / 32 ≈ 125 个时间步
        self.conv4 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=4, padding=2),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
        )
        # Mamba 用于建模长程时间依赖
        # d_state / d_conv / expand 可按需调整
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        # 把可变长度 (约 125) 压缩到固定的 75 帧
        self.pool = nn.AdaptiveAvgPool1d(target_frames)
        # 最终线性卷积映射到 EnCodec latent 维度
        self.out_proj = nn.Conv1d(d_model, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG 输入，形状 [batch, channels, time]，其中 time=4000 (8s×500Hz)
        Returns:
            形状 [batch, latent_dim, target_frames] 的隐藏表示，默认 latent_dim=128, target_frames=75
        """
        # 卷积前端：下采样时间维度
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x 现在形状 [batch, d_model, sequence_len≈125]
        # 转换为 [batch, sequence_len, d_model] 供 Mamba 使用
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        # 某些 Mamba 实现可能返回 (output, new_state)，取第一个
        if isinstance(x, tuple):
            x = x[0]
        # 转回卷积布局 [batch, d_model, sequence_len]
        x = x.permute(0, 2, 1)
        # 调整时间长度到 75 帧
        x = self.pool(x)
        # 投影到 latent_dim (=128)
        x = self.out_proj(x)
        return x


if __name__ == "__main__":
    # 创建模型
    model = EEGEncoder()
    # 假设有批量大小为 4 的 EEG 数据，32 通道，4000 时间采样点
    eeg_input = torch.randn(4, 32, 4000)
    # 前向传播得到 EnCodec latent 形状的输出
    latent = model(eeg_input)
    print(latent.shape)  # -> torch.Size([4, 128, 75])
