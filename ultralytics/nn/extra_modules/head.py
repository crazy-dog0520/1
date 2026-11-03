import math
import torch
import torch.nn as nn
from ..modules import Conv, DFL
from .block import *
from ultralytics.utils.tal import dist2bbox, make_anchors
# from ultralytics.utils.ops import nmsfree_postprocess

__all__ = ['SCBN', 'TDSCBN', 'FSCBN']
#用一个可学习的标量，对整个特征图做缩放。
class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
'''
把三个尺度的特征图转换成检测结果（边界框坐标 + 类别概率）
训练模式：特征图 → 原始预测值（供Loss计算）
推理模式：特征图 → 解码后的真实坐标+概率（供NMS筛选）
'''
class SCBN(nn.Module):
    # Lightweight Shared Convolutional Separate BN Detection Head

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor 每个 anchor 的总输出 = 分类 + 边界框
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(nn.Conv2d(hidc, hidc, 3, 1, 1), nn.Conv2d(hidc, hidc, 3, 1, 1))
        self.separate_bn = nn.ModuleList(nn.Sequential(nn.BatchNorm2d(hidc), nn.BatchNorm2d(hidc)) for _ in ch)
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    #把三个尺度的特征图，分别通过检测头，输出原始预测值，供 Loss 函数计算。
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        ##把三个尺度的特征图，分别通过检测头，输出原始预测值，供 Loss 函数计算。
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            for j in range(len(self.share_conv)):
                x[i] = self.act(self.separate_bn[i][j](self.share_conv[j](x[i])))  # Fix: [i][j] 每个尺度用自己独立的BN
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path 把三个尺度的原始预测值合并，生成 anchor 坐标，分离框和类别，解码出真实边界框坐标。
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)
        #特殊设备适配,组装最终输出,按场景返回
        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box    这是一个"中性"的初始框大小,不偏向大框也不偏向小框,网络从中性状态开始学习，不会一开始就极端偏斜
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)  bias = log(先验概率),目标：让初始类别预测匹配真实目标密度,必须精确，否则初始 Loss 爆炸
        #5 / m.nc / (640 / 16) ** 2每个位置、每个类别有目标的先验概率
    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    #把网络输出的原始分布预测（64维）转换成真实的边界框坐标（4维）。


class TDSCBN(nn.Module):
    """
    Task-Decoupled Shared Convolutional BN Detection Head (任务解耦共享卷积检测头)

    在原始 SCBN 基础上，在共享卷积之后加入任务专用通道注意力：
      - cls_attn[i]：分类专用 SE 注意力，引导特征关注外观/纹理通道
      - reg_attn[i]：回归专用 SE 注意力，引导特征关注边缘/空间通道
    每个检测尺度（P3/P4/P5）拥有独立的注意力权重。

    动机：SCBN 的 cv2（回归）和 cv3（分类）共享同一份特征，
         但两个任务对特征的需求不同，导致特征表示相互妥协。
         TDSCBN 通过任务解耦注意力，使每个任务获得各自最优的特征表示，
         提升分类准确率和定位精度的同时保持轻量化设计。

    新增参数：约 2 × nl × 2 × (hidc/8) × hidc ≈ 0.2M（极少）
    """
    dynamic = False
    export  = False
    shape   = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc      = nc
        self.nl      = len(ch)
        self.reg_max = 16
        self.no      = nc + self.reg_max * 4
        self.stride  = torch.zeros(self.nl)

        # ── 继承 SCBN 主体结构 ──────────────────────────────────
        self.conv        = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        self.share_conv  = nn.Sequential(nn.Conv2d(hidc, hidc, 3, 1, 1),
                                          nn.Conv2d(hidc, hidc, 3, 1, 1))
        self.separate_bn = nn.ModuleList(
            nn.Sequential(nn.BatchNorm2d(hidc), nn.BatchNorm2d(hidc)) for _ in ch)
        self.act   = nn.SiLU()
        self.cv2   = nn.Conv2d(hidc, 4 * self.reg_max, 1)   # 回归头
        self.cv3   = nn.Conv2d(hidc, self.nc, 1)             # 分类头
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl   = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # ── 新增：任务解耦通道注意力（每尺度独立）─────────────────
        r = max(hidc // 8, 8)   # 压缩维度
        def _make_se():
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidc, r, 1, bias=False),
                nn.SiLU(),
                nn.Conv2d(r, hidc, 1, bias=False),
                nn.Sigmoid()
            )
        self.cls_attn = nn.ModuleList(_make_se() for _ in ch)  # 分类专用注意力
        self.reg_attn = nn.ModuleList(_make_se() for _ in ch)  # 回归专用注意力

    def forward(self, x):
        for i in range(self.nl):
            # 共享卷积 + 独立 BN（与 SCBN 相同）
            x[i] = self.conv[i](x[i])
            for j in range(len(self.share_conv)):
                x[i] = self.act(self.separate_bn[i][j](self.share_conv[j](x[i])))

            # ── 任务解耦：各自注意力加权后再送入对应预测头 ──
            cls_feat = x[i] * self.cls_attn[i](x[i])   # 分类专用特征
            reg_feat = x[i] * self.reg_attn[i](x[i])   # 回归专用特征

            x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)),
                               self.cv3(cls_feat)), 1)

        if self.training:
            return x

        # ── 推理路径（与 SCBN 完全相同）──────────────────────────
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]; img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h],
                                    device=box.device).reshape(1, 4, 1)
            norm  = self.strides / (self.stride[0] * img_size)
            dbox  = dist2bbox(self.dfl(box) * norm,
                              self.anchors.unsqueeze(0) * norm[:, :2],
                              xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """初始化偏置（与 SCBN 相同）"""
        m = self
        m.cv2.bias.data[:] = 1.0
        m.cv3.bias.data[:m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes),
                         self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


class FSCBN(nn.Module):
    """
    Full Separate BN Detection Head（全独立批归一化检测头）

    原始 SCBN 的设计哲学：共享卷积权重 + 独立 BN（保留尺度特异性）
    但这一设计只应用在 share_conv 阶段，预测层 cv2/cv3 仍使用完全共享的权重和归一化。

    FSCBN 将独立 BN 延伸至预测层：
      cv2（回归头）：共享卷积权重 + 每尺度独立 BN → 尺度特异性回归归一化
      cv3（分类头）：共享卷积权重 + 每尺度独立 BN → 尺度特异性分类归一化

    动机：P3（80×80，小目标）和 P5（20×20，大目标）的预测输出统计特性不同，
         用同一套归一化参数是次优的。独立 BN 允许每个尺度独立调整预测分布，
         小目标和大目标各自获得最优的归一化状态。

    参数增量：约 2 × nl × (4×reg_max + nc) 个 BN 参数（极少）
    """
    dynamic = False
    export  = False
    shape   = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc      = nc
        self.nl      = len(ch)
        self.reg_max = 16
        self.no      = nc + self.reg_max * 4
        self.stride  = torch.zeros(self.nl)

        # ── 继承 SCBN 主体结构（完全相同）──────────────────────────
        self.conv        = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        self.share_conv  = nn.Sequential(nn.Conv2d(hidc, hidc, 3, 1, 1),
                                          nn.Conv2d(hidc, hidc, 3, 1, 1))
        self.separate_bn = nn.ModuleList(
            nn.Sequential(nn.BatchNorm2d(hidc), nn.BatchNorm2d(hidc)) for _ in ch)
        self.act   = nn.SiLU()
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl   = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # ── 改进：预测层共享权重 + 每尺度独立 BN ────────────────────
        # cv2：回归头（共享卷积，无 bias，由独立 BN 提供偏置）
        self.cv2     = nn.Conv2d(hidc, 4 * self.reg_max, 1, bias=False)
        self.cv2_bn  = nn.ModuleList(nn.BatchNorm2d(4 * self.reg_max) for _ in ch)

        # cv3：分类头（共享卷积，无 bias，由独立 BN 提供偏置）
        self.cv3     = nn.Conv2d(hidc, self.nc, 1, bias=False)
        self.cv3_bn  = nn.ModuleList(nn.BatchNorm2d(self.nc) for _ in ch)

    def forward(self, x):
        for i in range(self.nl):
            # 共享卷积 + 独立 BN（与 SCBN 相同）
            x[i] = self.conv[i](x[i])
            for j in range(len(self.share_conv)):
                x[i] = self.act(self.separate_bn[i][j](self.share_conv[j](x[i])))

            # ── 改进：预测层也用独立 BN ──
            reg = self.cv2_bn[i](self.cv2(x[i]))   # 尺度专属回归归一化
            cls = self.cv3_bn[i](self.cv3(x[i]))   # 尺度专属分类归一化
            x[i] = torch.cat((self.scale[i](reg), cls), 1)

        if self.training:
            return x

        # ── 推理路径（与 SCBN 完全相同）──────────────────────────────
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]; img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h],
                                    device=box.device).reshape(1, 4, 1)
            norm  = self.strides / (self.stride[0] * img_size)
            dbox  = dist2bbox(self.dfl(box) * norm,
                              self.anchors.unsqueeze(0) * norm[:, :2],
                              xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """初始化偏置（BN 的 bias 替代 conv bias）"""
        m = self
        # cv2_bn bias 初始化为中性值
        for bn in m.cv2_bn:
            nn.init.constant_(bn.bias, 1.0)
        # cv3_bn bias 初始化为先验概率对应值
        for bn in m.cv3_bn:
            nn.init.constant_(bn.bias, math.log(5 / m.nc / (640 / 16) ** 2))

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes),
                         self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


    '''
    为什么不直接预测一个距离值？
    直接预测：输出 2.7（一个数）
    分布预测：输出 [0,0,0.3,0.7,0,...,0]（16个概率）
    
    分布预测更稳定：
        目标框边界往往不确定（模糊、遮挡）
        用概率分布表达"大概在2~3格之间"
        比强行预测一个精确值更符合实际
    '''
