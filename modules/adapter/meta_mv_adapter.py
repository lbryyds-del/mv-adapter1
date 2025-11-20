# models/meta_mv_adapter.py
import math
import torch
from torch import nn
import torch.nn.functional as F

class TemporalGatedCrossModalMetaAdapter(nn.Module):
    """
    输入:
      w:   [C, D]           # 每类文本原型
      F:   [C, KT, D]       # 每类支持视频帧级特征堆叠
      Et:  [C, KT, De] or None  # 每帧时间嵌入(来自 TA 隐态或位置编码，已映射至 D 或 De)
      fC:  [C, Df] 或 [Df] 或 None  # 跨模态共享因子(真实或降级)
    输出:
      w_hat: [C, D]         # 细化后的时序感知类别原型
    """
    def __init__(self, dim, de=0, df=0, heads=4, proj_hidden=256, vec_gate=True):
        super().__init__()
        self.dim = dim
        self.de = de
        self.df = df
        self.heads = heads
        self.dk = dim

        in_dim_kv = dim + (de if de > 0 else 0)
        self.W1 = nn.Linear(in_dim_kv, self.dk, bias=False)  # K/V 投影
        self.W2 = nn.Linear(dim, self.dk, bias=False)        # Q 投影
        self.Wv = nn.Linear(in_dim_kv, dim, bias=False)      # V -> D

        # fC 调制（等效 W1'/W2' 的加性注入）
        self.fC_proj = nn.Linear(df, self.dk, bias=False) if df > 0 else None
        self.alpha = nn.Parameter(torch.ones(1)*0.1)
        self.beta  = nn.Parameter(torch.ones(1)*0.1)

        # 门控 g(w, Et)（逐维门控更稳）
        gate_in = dim + (de if de > 0 else 0)
        if vec_gate:
            self.gate = nn.Sequential(
                nn.Linear(gate_in, proj_hidden), nn.ReLU(True),
                nn.Linear(proj_hidden, dim), nn.Sigmoid()
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(gate_in, proj_hidden), nn.ReLU(True),
                nn.Linear(proj_hidden, 1), nn.Sigmoid()
            )
        self.vec_gate = vec_gate

        # 稳定训练：门控偏置初始化为负
        if hasattr(self.gate[-1], "bias"):
            with torch.no_grad():
                self.gate[-1].bias.fill_(-2.0)

        for m in [self.W1, self.W2, self.Wv]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, w, F, Et=None, fC=None):
        """
        w:  [C,D], F:[C,KT,D], Et:[C,KT,De] or None, fC:[C,Df] or [Df] or None
        """
        C, KT, D = F.shape

        # 1) KV 侧拼时间嵌入
        Fp = torch.cat([F, Et], dim=-1) if (Et is not None) else F  # [C,KT,D(+De)]

        # 2) 线性投影
        K = self.W1(Fp)          # [C,KT,dk]
        Q = self.W2(w)           # [C,dk]
        V = self.Wv(Fp)          # [C,KT,D]

        # 3) fC 调制（等效 W1'/W2'）
        if (self.fC_proj is not None) and (fC is not None):
            if fC.dim() == 1:  # [Df] -> [C, Df]
                fC = fC.unsqueeze(0).expand(C, -1)
            fC_proj = self.fC_proj(fC)  # [C,dk]
            K = K + self.alpha * fC_proj.unsqueeze(1)   # [C,1,dk] 广播到KT
            Q = Q + self.beta  * fC_proj               # [C,dk]

        # 4) 多头注意
        dk_h = self.dk // self.heads
        Qh = Q.view(C, self.heads, dk_h).unsqueeze(2)                 # [C,h,1,dk_h]
        Kh = K.view(C, KT, self.heads, dk_h).permute(0,2,1,3)         # [C,h,KT,dk_h]
        Vh = V.view(C, KT, self.heads, D//self.heads).permute(0,2,1,3)# [C,h,KT,D/h]

        attn = (Qh @ Kh.transpose(-1,-2)) / math.sqrt(dk_h)           # [C,h,1,KT]
        attn = attn.softmax(dim=-1)
        Ah = attn @ Vh                                                # [C,h,1,D/h]
        A  = Ah.permute(0,2,1,3).reshape(C, 1, D).squeeze(1)          # [C,D]

        # 5) 门控残差
        if Et is not None:
            Et_mean = Et.mean(dim=1)               # [C,De]
            gin = torch.cat([w, Et_mean], dim=-1)  # [C,D+De]
        else:
            gin = w                                 # [C,D]
        g = self.gate(gin)                          # [C,D] 或 [C,1]
        if not self.vec_gate:
            g = g.expand_as(w)

        w_hat = F.normalize(w + g*A, p=2, dim=-1)   # [C,D]
        return w_hat
