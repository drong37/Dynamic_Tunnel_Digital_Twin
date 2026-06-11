
import cv2
import numpy as np
import torch
from time import time
import config
import traceback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VehicleTracker")

try:
    from strong_sort import StrongSORT
    from strong_sort.utils.parser import get_config
    STRONGSORT_AVAILABLE = True
    logger.info("StrongSORT库导入成功")
except Exception as e:
    STRONGSORT_AVAILABLE = False
    logger.warning(f"警告: StrongSORT库导入失败: {e}")

try:
    from vehicle_speed_calculator import VehicleSpeedCalculator
    SPEED_CALCULATOR_AVAILABLE = True
    logger.info("速度计算器导入成功")
except Exception as e:
    SPEED_CALCULATOR_AVAILABLE = False
    logger.warning(f"警告: 速度计算器导入失败: {e}")

class VehicleTracker:
    
    def __init__(self, camera_id, strongsort_config_path=None, strongsort_weights=None):
        self.camera_id = camera_id
        
        if STRONGSORT_AVAILABLE:
            try:
                cfg_path = strongsort_config_path or config.STRONGSORT_CONFIG_PATH
                weights = strongsort_weights or config.STRONGSORT_WEIGHTS
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                cfg = get_config()
                cfg.merge_from_file(cfg_path)
                
                self.tracker = StrongSORT(
                    model_weights=weights,
                    device=device,
                    fp16=False,
                    max_dist=cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.STRONGSORT.MAX_AGE,
                    n_init=cfg.STRONGSORT.N_INIT,
                    nn_budget=cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=cfg.STRONGSORT.EMA_ALPHA
                )
                
                logger.info(f"StrongSORT跟踪器初始化成功，摄像头ID: {camera_id}，设备: {device}")
            except Exception as e:
                logger.error(f"警告: StrongSORT初始化失败: {e}")
                self.tracker = None
                traceback.print_exc()
        else:
            logger.warning(f"StrongSORT不可用，跟踪功能将受限")
            self.tracker = None
        
        self.frame_count = 0
        self.total_tracking_time = 0
        
        self.id_colors = {}
        
        self.detection_region = None
        if camera_id in config.DETECTION_REGIONS:
            self.detection_region = np.array(config.DETECTION_REGIONS[camera_id], dtype=np.int32)
            logger.info(f"加载摄像头 {camera_id} 的检测区域: {len(self.detection_region)} 个点")
        
        self.total_detections = 0
        self.region_filtered = 0
    
    def update(self, frame, detections):
        if not detections:
            return []
        
        self.total_detections += len(detections)
        
        start_time = time()

        filtered_detections = detections
        
        bbox_xywh = []
        confidences = []
        classes = []
        
        for det in filtered_detections:
            x1, y1, x2, y2, conf, cls_id = det
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w/2
            cy = y1 + h/2
            
            bbox_xywh.append([cx, cy, w, h])
            confidences.append(conf)
            classes.append(cls_id)
            
        if not bbox_xywh:
            return []
        
        bbox_xywh = np.array(bbox_xywh, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        
        classes = torch.tensor([int(cls) for cls in classes], dtype=torch.int64)
        
        if self.tracker is not None and len(bbox_xywh) > 0:
            try:
                with torch.no_grad():
                    outputs = self.tracker.update(bbox_xywh, confidences, classes, frame)
            except Exception as e:
                logger.error(f"StrongSORT跟踪出错: {e}")
                traceback.print_exc()
                outputs = np.empty((0, 7))
        else:
            outputs = np.empty((0, 7))
        
        tracking_time = time() - start_time
        self.total_tracking_time += tracking_time
        self.frame_count += 1
        
        tracked_vehicles = []
        
        for output in outputs:
            try:
                x1, y1, x2, y2, track_id, class_id, conf = output
                
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                
                bottom_center = [int((x1 + x2) / 2), int(y2)]
                
                in_detection_region = False
                if self.detection_region is not None:
                    in_detection_region = cv2.pointPolygonTest(self.detection_region, tuple(bottom_center), False) >= 0
                
                vehicle_patch = frame[y1:y2, x1:x2].copy() if 0 <= y1 < y2 and 0 <= x1 < x2 else None
                
                vehicle_info = {
                    'camera_id': self.camera_id,
                    'local_track_id': int(track_id),
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int(class_id),
                    'confidence': float(conf),
                    'patch': vehicle_patch,
                    'timestamp': time(),
                    'in_detection_region': in_detection_region
                }
                
                tracked_vehicles.append(vehicle_info)
            except Exception as e:
                logger.error(f"处理跟踪结果出错: {e}")

        return tracked_vehicles
    
    def draw_tracks(self, frame, tracked_vehicles, global_ids=None):
        annotated_frame = frame.copy()
        
        # if self.detection_region is not None:
        #     overlay = annotated_frame.copy()
        #     cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
            
        #     cv2.polylines(annotated_frame, [self.detection_region], True, (0, 255, 0), 2)
        
        if global_ids is None:
            global_ids = {}
        
        for vehicle in tracked_vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            local_id = vehicle['local_track_id']
            cls_id = vehicle['class_id']
            in_region = vehicle.get('in_detection_region', True)
            
            global_id = global_ids.get(local_id, local_id)
            
            color = self._get_color_by_id(global_id)
            
            if in_region:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            else:
                self._draw_dashed_rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            class_name = self._get_class_name(cls_id)
            
            region_mark = "IN" if in_region else "OUT"
            speed_text = f", {vehicle.get('speed', 0):.1f} km/h" if 'speed' in vehicle and vehicle['speed'] > 0 else ""
            track_label = f"ID:{global_id} {class_name} ({region_mark}{speed_text})"
            # track_label = f"ID:{global_id} {class_name} ({region_mark})"
            
            text_size = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, 
                         (x1, y1-text_size[1]-5), 
                         (x1+text_size[0]+5, y1), 
                         color, -1)
            
            cv2.putText(annotated_frame, track_label, (x1+2, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            bottom_center = (int((x1 + x2) / 2), int(y2))
            center_color = (0, 255, 0) if in_region else (0, 0, 255)
            cv2.circle(annotated_frame, bottom_center, 3, center_color, -1)
        
        camera_label = f"Camera: {self.camera_id}"
        cv2.putText(annotated_frame, camera_label, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.frame_count > 0:
            avg_tracking_time = self.total_tracking_time / self.frame_count
            tracking_info = f"Tracking: {avg_tracking_time*1000:.1f}ms | In/Out: {len([v for v in tracked_vehicles if v.get('in_detection_region', True)])}/{len([v for v in tracked_vehicles if not v.get('in_detection_region', True)])}"
            cv2.putText(annotated_frame, tracking_info, (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return annotated_frame
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=1):
        x1, y1 = pt1
        x2, y2 = pt2
        
        self._draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness)
        self._draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness)
        self._draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness)
        self._draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness)
    
    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=5, gap_length=5):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        dashes = int(dist / (dash_length + gap_length))
        ratio = dash_length / (dash_length + gap_length)
        
        for i in range(dashes):
            start_ratio = i / dashes
            end_ratio = min((i + ratio) / dashes, 1.0)
            
            start_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
                       int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
            
            end_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
                     int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))
            
            cv2.line(img, start_pt, end_pt, color, thickness)
    
    def _get_color_by_id(self, idx):
        idx = abs(int(idx))
        
        if idx in self.id_colors:
            return self.id_colors[idx]
        
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0),
            (128, 0, 128),
            (0, 128, 128),
            (180, 105, 255),
            (50, 205, 50),
            (40, 110, 170),
            (238, 130, 238),
            (152, 251, 152),
            (135, 206, 235),
            (255, 165, 0),
            (220, 20, 60),
        ]
        
        if idx < len(colors):
            color = colors[idx % len(colors)]
        else:
            r = (idx * 127) % 256
            g = (idx * 91) % 256
            b = (idx * 47) % 256
            color = (b, g, r)
        
        self.id_colors[idx] = color
        return color
    
    def _get_class_name(self, class_id):
        class_names = {
            2: "Car",
            5: "Bus",
            7: "Truck"
        }
        return class_names.get(class_id, f"Class {class_id}")


if __name__ == "__main__":
    from vehicle_detector import VehicleDetector
    
    detector = VehicleDetector()
    tracker = VehicleTracker(camera_id='cam1', strongsort_config_path=config.STRONGSORT_CONFIG_PATH, strongsort_weights=config.STRONGSORT_WEIGHTS)
    
    video_path = '/home/rongd/multicamera_vehicle/data/input_videos/1.mp4'
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        
        tracked_vehicles = tracker.update(frame, detections)
        
        global_ids = {v['local_track_id']: v['local_track_id'] + 1000 for v in tracked_vehicles}
        
        result_frame = tracker.draw_tracks(frame, tracked_vehicles, global_ids)
        
        cv2.imshow('Vehicle Tracking', result_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()