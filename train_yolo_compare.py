import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

DATA    = r'D:\Deep_learning\StarEye improve\mydata\data.yaml'
EPOCHS  = 100
IMGSZ   = 640
BATCH   = 8
WORKERS = 0
DEVICE  = '0'

if __name__ == '__main__':

    # ── YOLOv8n ───────────────────────────────────────────────────
    print('=' * 60)
    print('Training: YOLOv8n (comparison)')
    print('=' * 60)
    model = YOLO('yolov8n.pt')
    model.train(
        data         = DATA,
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        batch        = BATCH,
        workers      = WORKERS,
        device       = DEVICE,
        optimizer    = 'SGD',
        close_mosaic = 0,
        cache        = False,
        project      = 'runs/compare',
        name         = 'yolov8n',
        exist_ok     = True,
    )

    # ── YOLOv8s ───────────────────────────────────────────────────
    print('=' * 60)
    print('Training: YOLOv8s (comparison)')
    print('=' * 60)
    model = YOLO('yolov8s.pt')
    model.train(
        data         = DATA,
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        batch        = BATCH,
        workers      = WORKERS,
        device       = DEVICE,
        optimizer    = 'SGD',
        close_mosaic = 0,
        cache        = False,
        project      = 'runs/compare',
        name         = 'yolov8s',
        exist_ok     = True,
    )

    print('=' * 60)
    print('完成！结果保存在 runs/compare/')
    print('=' * 60)
