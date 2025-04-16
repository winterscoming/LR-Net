import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-p2.yaml')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset/VisDrone.yaml',
                # cfg='ultralytics/cfg/exp1.yaml',
                cache=False,
                project='runs/train',
                name='exp',
                epochs=300,
                batch=8,
                close_mosaic=10,
                optimizer='SGD', # using SGD
                workers=2,
                patience=1000,
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2
                )