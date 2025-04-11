
from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8m.pt')  # load an official model
    model = YOLO('runs/detect/train2/weights/best.pt')  # load a custom model

    print(model)
    # Validate the model
    metrics = model.val(split='val')  # no arguments needed, dataset and settings remembered
    print(model)
