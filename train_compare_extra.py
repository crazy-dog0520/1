import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, r'D:\Deep_learning\StarEye improve')
from ultralytics import YOLO, RTDETR

DATA    = r'D:\Deep_learning\StarEye improve\mydata\data.yaml'
EPOCHS  = 100
IMGSZ   = 640
WORKERS = 0
DEVICE  = '0'

if __name__ == '__main__':

    # ── YOLOv8m (pretrained) ──────────────────────────────────────────
    # 参数量 ~25M，介于 YOLOv8s 和 YOLOv8l 之间
    # 对比价值：展示 StarEye 以极少参数接近中等模型精度
    print('=' * 60)
    print('Training: YOLOv8m (pretrained)')
    print('=' * 60)
    model = YOLO('yolov8m.pt')
    model.train(
        data         = DATA,
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        batch        = 8,
        workers      = WORKERS,
        device       = DEVICE,
        optimizer    = 'SGD',
        close_mosaic = 0,
        cache        = False,
        project      = 'runs/compare',
        name         = 'yolov8m',
        exist_ok     = True,
    )
    print('YOLOv8m 完成！\n')

    # ── RT-DETR-l (pretrained) ────────────────────────────────────────
    # 基于 Transformer 的实时检测器，代表不同技术路线
    # 对比价值：展示 StarEye 与 Transformer 方法的竞争力
    print('=' * 60)
    print('Training: RT-DETR-l (pretrained)')
    print('=' * 60)
    model = RTDETR('rtdetr-l.pt')
    model.train(
        data         = DATA,
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        batch        = 4,       # RT-DETR 显存占用大，batch 减半
        workers      = WORKERS,
        device       = DEVICE,
        close_mosaic = 0,
        cache        = False,
        project      = 'runs/compare',
        name         = 'rtdetr_l',
        exist_ok     = True,
    )
    print('RT-DETR-l 完成！\n')

    print('=' * 60)
    print('全部完成！结果保存在 runs/compare/')
    print('=' * 60)
