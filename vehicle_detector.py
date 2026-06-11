
import cv2
import torch
import numpy as np
from time import time
from ultralytics import YOLO
import config

class VehicleDetector:
    
    def __init__(self, model_path=None, conf_threshold=0.35, vehicle_classes=None):
        if model_path is None:
            model_path = config.YOLO_MODEL_PATH
            
        self.conf_threshold = conf_threshold
        self.vehicle_classes = vehicle_classes or config.VEHICLE_CLASSES
        
        self.model = YOLO(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Vehicle detector using device: {self.device}")
        
        self.frame_count = 0
        self.total_inference_time = 0
    
    def detect(self, frame):
        start_time = time()
        
        results = self.model(frame, stream=True, device=self.device, verbose=False, imgsz=960, iou=0.5, agnostic_nms=True)
        
        inference_time = time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in self.vehicle_classes and conf > self.conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    
                    detections.append([x1, y1, x2, y2, conf, cls_id])
        
        return detections
    
    def draw_detections(self, frame, detections):
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            
            if cls_id == 2:
                color = (0, 255, 0)
            elif cls_id == 5:
                color = (255, 0, 0)
            elif cls_id == 7:
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{self._get_class_name(cls_id)}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if self.frame_count > 0:
            avg_time = self.total_inference_time / self.frame_count
            fps = 1.0 / max(avg_time, 1e-6)
            cv2.putText(annotated_frame, f"Inference: {avg_time*1000:.1f}ms FPS: {fps:.1f}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return annotated_frame
    
    def _get_class_name(self, class_id):
        class_names = {
            2: "Car",
            5: "Bus",
            7: "Truck"
        }
        return class_names.get(class_id, f"Class {class_id}")


if __name__ == "__main__":
    detector = VehicleDetector()
    
    video_path = '/home/rongd/multicamera_vehicle/data/input_videos/同向_起点.mp4'
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        
        result_frame = detector.draw_detections(frame, detections)
        
        cv2.imshow('Vehicle Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
