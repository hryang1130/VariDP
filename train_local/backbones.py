"""backbones.py — DP 的三种噪声预测主干，统一接口 forward(x, t_emb, c) -> (B, Tp, A)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNoisePred(nn.Module):
    """原 dp_lib 的实现：展平拼接 -> MLP（baseline，行为与旧版完全一致）"""

    def __init__(self, Tp, act_dim, cond_dim, t_dim=128, hidden=256, n_layers=3):
        super().__init__()
        self.Tp, self.act_dim = Tp, act_dim
        dims = [Tp * act_dim + t_dim + cond_dim] + [hidden] * n_layers + [Tp * act_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.Mish(), nn.LayerNorm(dims[i + 1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, c):
        b = x.shape[0]
        return self.net(torch.cat([x.reshape(b, -1), t, c], dim=-1)
                        ).reshape(b, self.Tp, self.act_dim)


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
            nn.GroupNorm(8, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    """Conv1d 残差块 + 时间步加法注入 + FiLM 条件调制"""

    def __init__(self, in_ch, out_ch, t_dim, cond_dim, k=5):
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch, k)
        self.block2 = Conv1dBlock(out_ch, out_ch, k)
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(t_dim, out_ch))
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t, c):
        s, b = self.film(c).chunk(2, dim=-1)          # (B, out_ch)
        h = self.block1(x) + self.time_mlp(t)[:, :, None]
        h = h * s[:, :, None] + b[:, :, None]          # FiLM
        h = self.block2(h)
        return h + self.res(x)


class ConditionalUNet1D(nn.Module):
    """最小版 1D 时序 UNet（对齐 DP-CNN 主干：下采样-上采样 + skip + FiLM）"""

    def __init__(self, Tp, act_dim, cond_dim, t_dim=128, down_dims=(64, 128, 256), k=5):
        super().__init__()
        assert Tp % (2 ** (len(down_dims) - 1)) == 0, "Tp 必须能被下采样次数整除"
        self.Tp, self.act_dim = Tp, act_dim
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(t_dim, t_dim))
        self.conv_in = nn.Conv1d(act_dim, down_dims[0], 1)

        self.down_blocks, self.downsample = nn.ModuleList(), nn.ModuleList()
        ch = down_dims[0]
        for i, out_ch in enumerate(down_dims):
            self.down_blocks.append(ResidualTemporalBlock(ch, out_ch, t_dim, cond_dim, k))
            ch = out_ch
            if i < len(down_dims) - 1:
                self.downsample.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))

        self.mid_block = ResidualTemporalBlock(ch, ch, t_dim, cond_dim, k)

        self.up_blocks = nn.ModuleList()
        for i in range(len(down_dims) - 1, -1, -1):
            out_ch = down_dims[max(i - 1, 0)]
            self.up_blocks.append(
                ResidualTemporalBlock(down_dims[i] * 2, out_ch, t_dim, cond_dim, k))

        self.conv_out = nn.Conv1d(down_dims[0], act_dim, 1)

    def forward(self, x, t, c):
        h = self.conv_in(x.transpose(1, 2))            # (B, A, Tp) -> (B, ch, Tp)
        t = self.time_mlp(t)
        skips = []
        for i, blk in enumerate(self.down_blocks):
            h = blk(h, t, c)
            skips.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
        h = self.mid_block(h, t, c)
        n = len(self.down_blocks)
        for j, blk in enumerate(self.up_blocks):
            s = skips[n - 1 - j]
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-1], mode="nearest")
            h = blk(torch.cat([h, s], dim=1), t, c)
        return self.conv_out(h).transpose(1, 2)        # -> (B, Tp, A)


class DiffusionTransformer(nn.Module):
    """GPT 式：条件/时间步 token 作前缀 + 因果注意力的动作 token（对齐 DP-Transformer 变体）"""

    def __init__(self, Tp, act_dim, cond_dim, t_dim=128,
                 d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.Tp, self.act_dim = Tp, act_dim
        self.act_in = nn.Linear(act_dim, d_model)
        self.cond_in = nn.Linear(cond_dim, d_model)
        self.t_in = nn.Linear(t_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, act_dim)

    def forward(self, x, t, c):
        seq = torch.cat([self.cond_in(c)[:, None, :],
                         self.t_in(t)[:, None, :],
                         self.act_in(x)], dim=1)       # (B, 2+Tp, d)
        L = seq.shape[1]
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(seq, mask=mask)
        return self.out(h[:, 2:, :])                   # (B, Tp, A)


BACKBONES = ("mlp", "unet", "transformer")


def build_noise_pred(backbone, Tp, act_dim, cond_dim, t_dim=128,
                     hidden=256, n_layers=3,                  # 仅 mlp 用
                     down_dims=(64, 128, 256),                # 仅 unet 用
                     d_model=256, transformer_layers=4, n_heads=4):   # 仅 transformer 用
    """按名字构造噪声预测主干（unet / transformer 的层宽在这里改）"""
    if backbone == "mlp":
        return MLPNoisePred(Tp, act_dim, cond_dim, t_dim,
                            hidden=hidden, n_layers=n_layers)
    if backbone == "unet":
        return ConditionalUNet1D(Tp, act_dim, cond_dim, t_dim, down_dims=down_dims)
    if backbone == "transformer":
        return DiffusionTransformer(Tp, act_dim, cond_dim, t_dim,
                                    d_model=d_model, n_layers=transformer_layers,
                                    n_heads=n_heads)
    raise ValueError(f"未知 backbone: {backbone}，可选 {BACKBONES}")
