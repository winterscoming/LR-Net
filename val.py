import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp6/weights/best.pt')
    # model = YOLO('yolov8n.pt')
    model.val(data='dataset/VisDrone.yaml',
                split='val',
                save_json=True, # if you need to cal coco metrice
                project='runs/val',
                name='exp',
                )