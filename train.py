import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# Official Parameters: https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO(r'D:\Deep_learning\StarEye\models\StarEye.yaml', task='detect')
    #model = YOLO(r'D:\Deep_learning\StarEye-main\models\StarEye.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'D:\Deep_learning\StarEye\mydata\data.yaml',
                cache=False,
                imgsz=640,
                epochs=350,
                batch=8,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # Universal checkpointing
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='realhumansssstar', # Output file name
                )