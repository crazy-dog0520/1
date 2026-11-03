import torch
import torch.nn as nn
from ..modules.conv import Conv
from ..modules.block import *
from .attention import *
from ultralytics.utils.torch_utils import make_divisible
from timm.models.layers import DropPath

__all__ = ['C3_Star', 'C2f_Star', 'C3_Star_CAA', 'C2f_Star_CAA', 'C2f_Star_DSCAA', 'C2f_Star_MSCAA', 'C2f_Star_MSCAA_S', 'C2f_Star_DMSCAA', 'PreMSCAA', 'SAFusion']

#自动计算 padding 的大小，使卷积输出尺寸 = 输入尺寸（same 卷积）。
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x
'''
Star Operation 之前插 CAA（不好）：
    输入还是低维特征（dim通道）
    注意力基于粗糙特征生成
    对细长结构的判断不准确

Star Operation 之后插 CAA（当前做法）：
    输入是高维交互特征（mlp_ratio×dim通道）
    经过二阶特征交互，细长目标的特征更突出
    此时的注意力判断更准确

类比：
    先把图像增强（Star Operation）
    再让专家（CAA）判断哪里重要
    比直接让专家看原图效果更好
    
Star Operation 让特征更丰富
CAA 让网络聚焦在目标上
共同作用 = 在水下复杂背景中精准检测人体
'''
class Star_Block_CAA(Star_Block):
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)
        
        self.attention = CAA(mlp_ratio * dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x

'''
用 C3 骨架 + Star_Block 替换内部：
C3 的 cv1/cv2 双路：一路深度提炼，一路保留原始
Star_Block 负责深度路的二阶特征提炼
cv3 最后融合两路
C3_Star 是过渡版本，探索 Star_Block 嵌入 C3 的可行性
C2f_Star_CAA 是最终版本，用在 StarEye 的 neck 里
'''
class C3_Star(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block(c_) for _ in range(n)))

class C2f_Star(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block(self.c) for _ in range(n))

class C3_Star_CAA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block_CAA(c_) for _ in range(n)))

class C2f_Star_CAA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_CAA(self.c) for _ in range(n))

# Star_Block_DSCAA：将 CAA 替换为 DSCAA（可变形条带注意力）
# 其余结构与 Star_Block_CAA 完全一致，便于消融对比
class Star_Block_DSCAA(Star_Block):
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)
        self.attention = DSCAA(mlp_ratio * dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x

class C2f_Star_DSCAA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_DSCAA(self.c) for _ in range(n))


class PreMSCAA(nn.Module):
    """Shape-preserving MSCAA used on a source branch before pyramid alignment."""

    def __init__(self, channels):
        super().__init__()
        self.attention = MSCAA(channels)

    def forward(self, x):
        return self.attention(x)


class Star_Block_MSCAA(Star_Block):
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)
        self.attention = MSCAA(mlp_ratio * dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x

class C2f_Star_MSCAA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_MSCAA(self.c) for _ in range(n))

class Star_Block_MSCAA_S(Star_Block):
    """单路小核（1×7）消融变体"""
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)
        self.attention = MSCAA_S(mlp_ratio * dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x

class C2f_Star_MSCAA_S(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_MSCAA_S(self.c) for _ in range(n))

class Star_Block_DMSCAA(Star_Block):
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)
        self.attention = DMSCAA(mlp_ratio * dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x

class C2f_Star_DMSCAA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_DMSCAA(self.c) for _ in range(n))

class SAFusion(nn.Module):
    """
    尺度感知特征融合模块 (Scale-Aware Feature Fusion)

    替换 FPN/PAN neck 中的标准 Concat，加入内容自适应通道门控权重。

    动机：标准 Concat 对高层语义特征与低层细节特征等权拼接，
         但水下小目标（鱼类）的关键信息集中在高分辨率细节层，
         被低分辨率语义层稀释后影响检测性能。
         本模块通过 SE 风格的通道注意力为每个通道动态分配融合权重，
         使模型自主学习"哪些通道的特征更有价值"。

    结构：Concat → GlobalAvgPool → FC↓ → SiLU → FC↑ → Sigmoid → channel-wise multiply

    参数量：约 2 × (ch_total // 8) × ch_total  （极少，不影响轻量化特性）
    """
    def __init__(self, ch_total, dimension=1, residual=False):
        super().__init__()
        self.d = dimension
        self.residual = residual
        r = max(ch_total // 8, 8)   # 压缩比，最小保留 8 个通道
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                    # 全局平均池化，获取全局上下文
            nn.Conv2d(ch_total, r, 1, bias=False),      # 通道压缩
            nn.SiLU(),                                  # 非线性激活
            nn.Conv2d(r, ch_total, 1, bias=False),      # 通道还原
            nn.Sigmoid()                                # 权重归一化到 (0,1)
        )

    def forward(self, x):
        """
        x: list of tensors（与 Concat 接口相同）
        流程：先 Concat，再用通道门控加权
        """
        cat = torch.cat(x, self.d)      # [B, ch_total, H, W]
        gated = cat * self.gate(cat)
        # Checkpoints saved before RSAFusion do not contain the residual flag.
        return cat + gated if getattr(self, "residual", False) else gated


'''
C2f 比 C3：
    更多特征被保留（每层输出都参与最终融合）
    梯度更畅通（每个Block直连输出）
    精度更高，YOLOv8 全面替换为 C2f
CAA 的作用：
    在 Star Operation 之后，对 mlp_ratio×dim 维特征做注意力
    用 1×11 + 11×1 条带卷积，增强细长目标（手臂、腿）
    压制水下背景噪声
有 CAA：更强，但多一些参数和计算量
无 CAA：更轻，适合对速度要求极高的场景
'''
