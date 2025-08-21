import torch
import torch.nn as nn

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
        def forward(self, x):  # [B, T, C]
            return x

class Bottleneck1D(nn.Module):
    """SEANet风格的瓶颈残差块：1x1 降维 -> 3x1 深度可分离卷积 -> 1x1 升维 + shortcut"""
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(channels, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False),  # depthwise
            nn.GroupNorm(1, hidden),
            nn.ELU(),
            nn.Conv1d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class DownBlock(nn.Module):
    """残差块 + 下采样卷积"""
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.res = Bottleneck1D(in_ch, hidden=max(in_ch // 2, 32))
        self.act = nn.ELU()
        # kernel 选用 2*stride+1 与 EnCodec 风格相近；GroupNorm 避免小批量不稳定
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=2*stride+1, stride=stride, padding=stride, bias=False),
            nn.GroupNorm(1, out_ch),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.act(x)
        x = self.down(x)
        return x

class EEGEncoderSEAMamba(nn.Module):
    """
    加强版：SEANet-like 卷积前端 + Mamba 序列建模
    输入:  [B, 32, 4000]   (8s @ 500 Hz, 32通道)
    输出:  [B, 128, 75]    (对齐 EnCodec 的 latent 管道)
    """
    def __init__(self,
                 in_channels: int = 32,
                 stem_channels: int = 64,
                 stage_channels=(64, 128, 256, 256),
                 stage_strides=(2, 2, 2, 5),   # 总下采样=40，4000/40≈100帧
                 d_model: int = 256,           # Mamba/投影前的模型宽度
                 latent_dim: int = 128,
                 target_frames: int = 75):
        super().__init__()

        # 1) 空间融合 stem：先把 32 通道混合为 stem_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),  # /2
            nn.GroupNorm(1, stem_channels),
            nn.ELU(),
        )

        # 2) 多阶段：每阶段 残差瓶颈块 + 下采样
        stages = []
        ch_in = stem_channels
        for ch_out, s in zip(stage_channels, stage_strides):
            stages.append(DownBlock(ch_in, ch_out, stride=s))
            ch_in = ch_out
        self.stages = nn.Sequential(*stages)  # 最终时间长度约 4000 / (2*2*2*5) = ~100

        # 3) 对齐到 d_model
        self.to_model_dim = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(ch_in, d_model, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, d_model),
        )

        # 4) Mamba 序列建模（在较短的 ~100 帧上做，全局依赖更轻松）
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

        # 5) 对齐时间帧数到 75（4000/40≈100 → 75）
        self.pool = nn.AdaptiveAvgPool1d(target_frames)

        # 6) 输出投影到 128 latent 维度
        self.out_proj = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(d_model, latent_dim, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 32, 4000]
        return: [B, 128, 75]
        """
        x = self.stem(x)         # [B, C, T/2]
        x = self.stages(x)       # [B, C', ~100]
        x = self.to_model_dim(x) # [B, d_model, ~100]

        # Mamba 期望 [B, T, C]
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        if isinstance(x, tuple):  # 某些实现会返回 (y, state)
            x = x[0]
        x = x.permute(0, 2, 1)    # 回到 [B, d_model, ~100]

        x = self.pool(x)          # [B, d_model, 75]
        x = self.out_proj(x)      # [B, 128, 75]
        return x

if __name__ == "__main__":
    B = 2
    eeg = torch.randn(B, 32, 4000)
    enc = EEGEncoderSEAMamba()
    y = enc(eeg)
    print("Output:", y.shape)  # torch.Size([2, 128, 75])
