from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 加载模型
    model = YOLO(model="ultralytics/cfg/models/v8/YOLO-ED.yaml")  # 从头开始构建新模型
    print(model)
    # Use the model
    results = model.train(data="VOC.yaml",patience=0, epochs=300, device='0', batch=8, seed=42,imgsz=640)  # 训练模
