import math
import torch
from torch import nn
from torchvision.ops import deform_conv2d as tvops_deform_conv2d
from ..modules.conv import Conv


__all__ = ['CAA', 'DSCAA', 'MSCAA', 'MSCAA_S', 'DMSCAA']
#用条带卷积生成注意力权重，告诉网络"哪些位置更值得关注"。专门为水下细长目标（手臂、腿、身体轮廓）设计。
#条带卷积识别出细长延伸结构 → 注意力权重高 → 特征保留
#背景区域 → 注意力权重低 → 特征压制
#条带卷积核参数初始值：Kaiming Uniform 随机，范围约 [-0.3, +0.3]
# CVPR2024
class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size = 11, v_kernel_size = 11) -> None:
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)   #水下图像噪点多（光线折射、水流扰动），先平滑，再提取注意力，更稳定，尺寸不变
        self.conv1 = Conv(ch, ch)  #标准卷积，每个输出通道看所有输入通道，整合通道信息，不同通道代表不同特征，conv1 学习它们之间的关系
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)  #水平条带卷积
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)  #垂直条带卷积
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x

'''
DSCAA: Deformable Strip Context Anchor Attention（可变形条带上下文锚点注意力）

动机：
    原始 CAA 的 h_conv/v_conv 采样点位置硬编码在水平/垂直直线上，
    无法对齐游泳者斜向或弯曲的肢体，注意力权重偏离实际目标区域。

改进：
    用可变形卷积（Deformable Conv, DCN）替换固定条带卷积。
    每个采样点的位置由"偏移生成器"根据当前特征图内容动态预测，
    使 11 个采样点可以沿肢体真实走向弯曲排列。

偏移生成器（offset_gen）：
    3×3 普通卷积，输出 2×k 个通道（每个采样点 Δrow + Δcol 两个偏移值）。
    权重和偏置全部初始化为 0 → 训练初期所有偏移 ≈ 0，
    等价于标准固定条带卷积，保证训练稳定性，再逐步学习自适应偏移。

可变形条带权重（h_weight / v_weight）：
    声明为 nn.Parameter，形状 (ch, 1, kH, kW)，depthwise，
    Kaiming Uniform 初始化，与原始 CAA 的条带卷积参数量相同。

forward 流程：
    feat = conv1(avg_pool(x))
    h_offset = h_offset_gen(feat)           → (N, 22, H, W)
    feat_h   = deform_conv_h(feat, h_offset) → 水平可变形条带
    v_offset = v_offset_gen(feat_h)          → (N, 22, H, W)，利用水平上下文
    feat_v   = deform_conv_v(feat_h, v_offset)→ 垂直可变形条带
    attn = Sigmoid(conv2(feat_v))
    return attn * x
'''
class DSCAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        self.ch = ch
        self.h_kernel_size = h_kernel_size
        self.v_kernel_size = v_kernel_size

        # 与原始 CAA 相同的前处理：平滑噪声 + 通道整合
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)

        # 偏移生成器（零初始化，训练初期等价于固定条带）
        # 水平条带：kH=1, kW=h_kernel_size → offset 通道数 = 2*1*h_kernel_size
        self.h_offset_gen = nn.Conv2d(ch, 2 * h_kernel_size, 3, 1, 1)
        nn.init.zeros_(self.h_offset_gen.weight)
        nn.init.zeros_(self.h_offset_gen.bias)
        # 垂直条带：kH=v_kernel_size, kW=1 → offset 通道数 = 2*v_kernel_size*1
        self.v_offset_gen = nn.Conv2d(ch, 2 * v_kernel_size, 3, 1, 1)
        nn.init.zeros_(self.v_offset_gen.weight)
        nn.init.zeros_(self.v_offset_gen.bias)

        # 可变形条带卷积权重（depthwise：weight[i] 只对第 i 个通道操作）
        # weight 形状 (ch, 1, kH, kW) → groups = ch，等价于原始 CAA 的 groups=ch
        self.h_weight = nn.Parameter(torch.empty(ch, 1, 1, h_kernel_size))
        self.v_weight = nn.Parameter(torch.empty(ch, 1, v_kernel_size, 1))
        nn.init.kaiming_uniform_(self.h_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_weight, a=math.sqrt(5))
        self.h_bias = nn.Parameter(torch.zeros(ch))
        self.v_bias = nn.Parameter(torch.zeros(ch))

        # 与原始 CAA 相同的后处理
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv1(self.avg_pool(x))

        # 水平可变形条带：偏移由 feat 生成，采样点沿水平方向自适应偏移
        h_offset = self.h_offset_gen(feat)                      # (N, 2*h_k, H, W)
        feat_h = tvops_deform_conv2d(
            feat, h_offset, self.h_weight, self.h_bias,
            stride=1,
            padding=(0, self.h_kernel_size // 2)                # same 输出尺寸
        )

        # 垂直可变形条带：偏移由 feat_h 生成，融合了水平方向的上下文信息
        v_offset = self.v_offset_gen(feat_h)                    # (N, 2*v_k, H, W)
        feat_v = tvops_deform_conv2d(
            feat_h, v_offset, self.v_weight, self.v_bias,
            stride=1,
            padding=(self.v_kernel_size // 2, 0)                # same 输出尺寸
        )

        attn = self.act(self.conv2(feat_v))
        return attn * x


'''
MSCAA: Multi-Scale Context Anchor Attention（多尺度条带上下文锚点注意力）

动机：
    原始 CAA 只用固定的 1×11+11×1 条带，感受野单一。
    水下目标距离不同，在特征图上的尺度也不同：
        近距离目标（大）→ 肢体跨度大，需要 1×11 长条带覆盖
        远距离目标（小）→ 肢体跨度小，1×11 引入过多背景噪声，1×7 更合适

设计：
    Branch-L：1×11 + 11×1，大感受野，对应近距离/大目标
    Branch-S：1×7  + 7×1，  小感受野，对应远距离/小目标
    Fusion：Concat(Branch-L, Branch-S) → 1×1 Conv 压缩回 ch 通道
    最终通过 Sigmoid 生成注意力权重，对输入 x 加权

相比原始 CAA：
    两路条带共享同一个 avg_pool + conv1 前处理（不重复计算）
    参数仅增加两组额外 strip conv + 一个 1×1 fusion Conv
    感受野从单一变为多尺度，对远近目标均有效
'''
class MSCAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        # 与原始 CAA 相同的前处理
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)

        # Branch-L：长条带，大感受野（对应近距离目标）
        self.h_conv_l = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv_l = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)

        # Branch-S：短条带，小感受野（对应远距离小目标）
        self.h_conv_s = nn.Conv2d(ch, ch, (1, 7), 1, (0, 3), 1, ch)
        self.v_conv_s = nn.Conv2d(ch, ch, (7, 1), 1, (3, 0), 1, ch)

        # 多尺度融合：Concat → 1×1 Conv 压回 ch 通道
        self.fusion = Conv(ch * 2, ch, 1)

        # 与原始 CAA 相同的后处理
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv1(self.avg_pool(x))          # 共享前处理

        out_l = self.v_conv_l(self.h_conv_l(feat))   # Branch-L：长条带
        out_s = self.v_conv_s(self.h_conv_s(feat))   # Branch-S：短条带

        out = self.fusion(torch.cat([out_l, out_s], dim=1))  # 多尺度融合
        attn = self.act(self.conv2(out))
        return attn * x


'''
DMSCAA: Dilated Multi-Scale Context Anchor Attention（膨胀多尺度条带注意力）

在 MSCAA（两路）基础上增加第三路膨胀条带：

三路设计：
    Branch-S：1×7,  dilation=1 → 实际覆盖  7 像素（远距离小目标）
    Branch-M：1×11, dilation=1 → 实际覆盖 11 像素（中距离目标，沿用原始CAA）
    Branch-L：1×11, dilation=2 → 实际覆盖 21 像素（近距离大目标/全身覆盖）

膨胀条带的 padding 计算：
    effective_k = dilation*(k-1)+1 = 2*10+1 = 21
    padding = (effective_k-1)//2 = 10
    参数量与普通 1×11 完全相同，感受野扩大一倍

Fusion：Concat(Branch-S, Branch-M, Branch-L) → Conv(ch*3, ch, 1)

三路覆盖了小/中/大三种尺度目标，同时 SCBN Bug 已修复，
检测头每个尺度使用真正独立的 BN，定位精度进一步提升。
'''
class MSCAA_S(nn.Module):
    """单路小核条带注意力（1×7），消融实验用：对比 CAA(1×11) 和 MSCAA(1×11+1×7)"""
    def __init__(self, ch):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1    = Conv(ch, ch)
        self.h_conv   = nn.Conv2d(ch, ch, (1, 7), 1, (0, 3), 1, ch)
        self.v_conv   = nn.Conv2d(ch, ch, (7, 1), 1, (3, 0), 1, ch)
        self.conv2    = Conv(ch, ch)
        self.act      = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv1(self.avg_pool(x))
        out  = self.v_conv(self.h_conv(feat))
        attn = self.act(self.conv2(out))
        return attn * x


class DMSCAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        # 与原始 CAA 相同的前处理
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)

        # Branch-S：短条带，dilation=1，覆盖 7 像素（远距离小目标）
        self.h_conv_s = nn.Conv2d(ch, ch, (1, 7), 1, (0, 3), 1, ch)
        self.v_conv_s = nn.Conv2d(ch, ch, (7, 1), 1, (3, 0), 1, ch)

        # Branch-M：标准条带，dilation=1，覆盖 11 像素（中距离目标）
        self.h_conv_m = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv_m = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)

        # Branch-L：膨胀条带，dilation=2，覆盖 21 像素（近距离大目标，参数量不变）
        # padding = dilation*(k-1)//2 = 2*5 = 10，保证输出尺寸不变
        self.h_conv_l = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, 10), dilation=(1, 2), groups=ch)
        self.v_conv_l = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (10, 0), dilation=(2, 1), groups=ch)

        # 三路融合
        self.fusion = Conv(ch * 3, ch, 1)

        # 与原始 CAA 相同的后处理
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv1(self.avg_pool(x))

        out_s = self.v_conv_s(self.h_conv_s(feat))   # Branch-S：7 像素
        out_m = self.v_conv_m(self.h_conv_m(feat))   # Branch-M：11 像素
        out_l = self.v_conv_l(self.h_conv_l(feat))   # Branch-L：21 像素（膨胀）

        out = self.fusion(torch.cat([out_s, out_m, out_l], dim=1))
        attn = self.act(self.conv2(out))
        return attn * x


'''
普通做法（YOLOv8原版）：
    torch.cat([fea1, fea2], dim=1)  → 通道数翻倍
    再用 Conv 压缩回原通道          → 有参数量开销
    参数量 = ch × 2ch × k × k（大）

Mix 的做法：
    只有 1 个可学习参数（w）
    直接加权求和，通道数不变
    参数量 = 1（极小）

代价：
    Concat+Conv 能学到复杂的特征融合方式
    Mix 只能学一个全局比例，表达能力较弱
    但极度轻量，适合追求速度的场景

w初始值为0.8，再通过Sigmoid    
不存 0.31，是因为直接存比例值会在边界处卡死梯度。
存 -0.80 再过 Sigmoid，让 w 在无界空间自由更新，
Sigmoid 保证输出始终合法，两全其美。
'''
#用一个可学习的参数，自动决定两个特征图如何混合的比例。
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)  #把普通张量升级为"可学习参数"
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out