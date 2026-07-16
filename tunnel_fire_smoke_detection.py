
import cv2
import numpy as np
import time
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ultralytics import YOLO
import config


@dataclass
class FireDetection:
    camera_id: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int  # 0: fire, 1: smoke
    class_name: str
    detection_time: float
    map_position: Tuple[int, int]
    real_position: Tuple[float, float] = (0.0, 0.0)
    

class TunnelFireSmokeDetector:
    
    def __init__(self, model_path: str = "models/yolo11n-fire-smoke-v2.pt", 
                 confidence_threshold: float = 0.5,
                 fire_icon_path: str = "utils/flammable.png"):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.fire_icon_path = fire_icon_path
        
        self.class_names = {0: "fire"}
        
        self._load_model()
        
        self._load_fire_icon()
        
        self.active_detections: Dict[str, List[FireDetection]] = {}  # camera_id -> detections
        self.detection_history: List[FireDetection] = []
        
        self.camera_calibrations: Dict[str, Any] = {}
        
        self.detection_timeout = 0.5
        
        print(f"火灾烟雾检测器初始化完成")
        print(f"模型路径: {self.model_path}")
        print(f"置信度阈值: {self.confidence_threshold}")
        print(f"火灾图标路径: {self.fire_icon_path}")
    
    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                print(f"警告: 模型文件不存在: {self.model_path}")
                print("请确保模型文件存在或检查路径")
                self.model = None
                return
            
            self.model = YOLO(self.model_path)
            print(f"YOLO模型加载成功: {self.model_path}")
            
            if hasattr(self.model, 'names'):
                model_classes = self.model.names
                print(f"模型支持的类别: {model_classes}")
            
        except Exception as e:
            print(f"加载YOLO模型失败: {e}")
            self.model = None
    
    def _load_fire_icon(self):
        try:
            if not os.path.exists(self.fire_icon_path):
                print(f"警告: 火灾图标文件不存在: {self.fire_icon_path}")
                self.fire_icon = self._create_default_fire_icon()
                self.fire_icon_size = (24, 24)
                return
            
            self.fire_icon = cv2.imread(self.fire_icon_path, cv2.IMREAD_UNCHANGED)
            if self.fire_icon is None:
                print(f"无法读取火灾图标: {self.fire_icon_path}")
                self.fire_icon = self._create_default_fire_icon()
                self.fire_icon_size = (24, 24)
                return

            self.fire_icon_size = (24, 24)
            self.fire_icon = cv2.resize(self.fire_icon, self.fire_icon_size)
            
            if self.fire_icon.shape[2] == 3:
                alpha_channel = np.ones((self.fire_icon.shape[0], self.fire_icon.shape[1], 1), dtype=np.uint8) * 255
                self.fire_icon = np.concatenate([self.fire_icon, alpha_channel], axis=2)
                print(f"为3通道图像添加了alpha通道")
            
            print(f"火灾图标加载成功: {self.fire_icon_path}, 大小: {self.fire_icon_size}, 通道数: {self.fire_icon.shape[2]}")
            
        except Exception as e:
            print(f"加载火灾图标失败: {e}")
            self.fire_icon = self._create_default_fire_icon()
            self.fire_icon_size = (24, 24)
    
    def _create_default_fire_icon(self) -> np.ndarray:
        size = 24
        icon = np.zeros((size, size, 4), dtype=np.uint8)
        
        center = (size // 2, size // 2)
        
        for y in range(size):
            for x in range(size):
                dx = x - center[0]
                dy = y - center[1] + 2
                
                if dy < 0:
                    flame_width = max(1, size // 4 + int(dy * 0.3))
                else:
                    flame_width = size // 3
                
                distance = np.sqrt(dx*dx + dy*dy)
                flame_radius = min(flame_width, size // 2 - 1)
                
                if distance <= flame_radius:
                    intensity = max(0, 1 - distance / flame_radius)
                    
                    if intensity > 0.7:
                        icon[y, x] = [0, 0, 255, int(255 * intensity)]
                    elif intensity > 0.4:
                        icon[y, x] = [0, 165, 255, int(255 * intensity)]
                    else:
                        icon[y, x] = [0, 255, 255, int(255 * intensity)]
        
        return icon
    
    def _get_camera_calibration(self, camera_id: str):
        if camera_id not in self.camera_calibrations:
            try:
                from camera_calibration import CameraCalibration
                self.camera_calibrations[camera_id] = CameraCalibration(camera_id)
            except Exception as e:
                print(f"无法加载摄像头 {camera_id} 的标定信息: {e}")
                self.camera_calibrations[camera_id] = None
        
        return self.camera_calibrations[camera_id]
    
    def detect_fire_smoke(self, frame: np.ndarray, camera_id: str) -> List[FireDetection]:
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False, imgsz=640)
            
            detections = []
            current_time = time.time()
            
            calibration = self._get_camera_calibration(camera_id)
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id not in self.class_names:
                        continue
                    
                    class_name = self.class_names[class_id]
                    
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    center = [center_x, center_y]

                    map_position = (0, 0)
                    real_position = (0.0, 0.0)

                    if calibration is not None:
                        try:
                            map_coords = calibration.map_to_ground(center)
                            map_position = (int(map_coords[0]), int(map_coords[1]))

                            import config
                            real_x, real_y = config.pixel_to_real_coordinates(map_coords[0], map_coords[1])
                            real_position = (real_x, real_y)
                        except Exception as e:
                            print(f"坐标映射失败 {camera_id}: {e}")

                    detection = FireDetection(
                        camera_id=camera_id,
                        bbox=bbox.tolist(),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                        detection_time=current_time,
                        map_position=map_position,
                        real_position=real_position
                    )
                    
                    detections.append(detection)
            
            self._update_active_detections(camera_id, detections, current_time)
            
            self.detection_history.extend(detections)
            
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-500:]

            return detections

        except Exception as e:
            print(f"火灾烟雾检测失败 {camera_id}: {e}")
            return []

    def detect_fire_smoke_batch(
        self,
        frames: List[np.ndarray],
        camera_ids: List[str]
    ) -> Dict[str, List[FireDetection]]:
        if self.model is None or len(frames) == 0:
            return {cam_id: [] for cam_id in camera_ids}

        try:
            results = self.model(frames, conf=self.confidence_threshold,
                                verbose=False, imgsz=640)

            current_time = time.time()
            detections_by_camera: Dict[str, List[FireDetection]] = {}

            for result, camera_id in zip(results, camera_ids):
                frame_detections = []
                calibration = self._get_camera_calibration(camera_id)

                if result.boxes is None:
                    detections_by_camera[camera_id] = []
                    continue

                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    if class_id not in self.class_names:
                        continue

                    center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
                    map_position = (0, 0)
                    real_position = (0.0, 0.0)

                    if calibration is not None:
                        try:
                            map_coords = calibration.map_to_ground(center)
                            map_position = (int(map_coords[0]), int(map_coords[1]))

                            real_x, real_y = config.pixel_to_real_coordinates(map_coords[0], map_coords[1])
                            real_position = (real_x, real_y)
                        except:
                            pass

                    detection = FireDetection(
                        camera_id=camera_id,
                        bbox=bbox.tolist(),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=self.class_names[class_id],
                        detection_time=current_time,
                        map_position=map_position,
                        real_position=real_position
                    )
                    frame_detections.append(detection)

                detections_by_camera[camera_id] = frame_detections
                self._update_active_detections(camera_id, frame_detections, current_time)
                self.detection_history.extend(frame_detections)

            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-500:]

            return detections_by_camera

        except Exception as e:
            print(f"批量火灾烟雾检测失败: {e}")
            return {cam_id: [] for cam_id in camera_ids}

    def _update_active_detections(self, camera_id: str, new_detections: List[FireDetection],
                                current_time: float):
        if camera_id not in self.active_detections:
            self.active_detections[camera_id] = []
        
        self.active_detections[camera_id] = [
            detection for detection in self.active_detections[camera_id]
            if current_time - detection.detection_time < self.detection_timeout
        ]
        
        self.active_detections[camera_id].extend(new_detections)
    
    def get_all_active_detections(self) -> List[FireDetection]:
        all_detections = []
        current_time = time.time()
        
        for camera_id, detections in self.active_detections.items():
            valid_detections = [
                detection for detection in detections
                if current_time - detection.detection_time < self.detection_timeout
            ]
            all_detections.extend(valid_detections)
        
        return all_detections
    
    def draw_detections_on_frame(self, frame: np.ndarray, detections: List[FireDetection]) -> np.ndarray:
        result_frame = frame.copy()
        
        for detection in detections:
            bbox = detection.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            if detection.class_id == 0:  # fire
                color = (0, 0, 255)
                label_bg_color = (0, 0, 255)
            else:  # smoke
                color = (128, 128, 128)
                label_bg_color = (128, 128, 128)
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_w, label_h = label_size
            
            cv2.rectangle(result_frame, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), label_bg_color, -1)
            
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            bottom_center_x = int((x1 + x2) / 2)
            bottom_center_y = y2
            cv2.circle(result_frame, (bottom_center_x, bottom_center_y), 5, color, -1)
            
            map_coord_text = f"Map: ({detection.map_position[0]}, {detection.map_position[1]})"
            cv2.putText(result_frame, map_coord_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_frame
    
    def draw_detections_on_map(self, map_image: np.ndarray) -> np.ndarray:
        result_map = map_image.copy()
        
        active_detections = self.get_all_active_detections()
        
        for detection in active_detections:
            map_x, map_y = detection.map_position
            
            if (0 <= map_x < map_image.shape[1] and 
                0 <= map_y < map_image.shape[0]):
                
                self._draw_transparent_icon(result_map, map_x, map_y)
                
                self._draw_fire_info_label_transparent(result_map, detection, map_x, map_y)
        
        return result_map
    
    def _draw_transparent_icon(self, map_image: np.ndarray, map_x: int, map_y: int):
        try:
            icon_h, icon_w = self.fire_icon_size
            
            icon_x1 = max(0, map_x - icon_w // 2)
            icon_y1 = max(0, map_y - icon_h // 2)
            icon_x2 = min(map_image.shape[1], icon_x1 + icon_w)
            icon_y2 = min(map_image.shape[0], icon_y1 + icon_h)
            
            actual_w = icon_x2 - icon_x1
            actual_h = icon_y2 - icon_y1
            
            if actual_w > 0 and actual_h > 0:
                resized_icon = cv2.resize(self.fire_icon, (actual_w, actual_h))
                
                map_roi = map_image[icon_y1:icon_y2, icon_x1:icon_x2]
                
                if resized_icon.shape[2] == 4:
                    icon_bgr = resized_icon[:, :, :3]
                    icon_alpha = resized_icon[:, :, 3:4] / 255.0
                    
                    blended = icon_bgr * icon_alpha + map_roi * (1 - icon_alpha)
                    map_image[icon_y1:icon_y2, icon_x1:icon_x2] = blended.astype(np.uint8)
                else:
                    map_image[icon_y1:icon_y2, icon_x1:icon_x2] = resized_icon
                    
        except Exception as e:
            print(f"绘制透明图标失败: {e}")
    
    def _draw_fire_info_label_transparent(self, map_image: np.ndarray, detection: FireDetection, 
                                        map_x: int, map_y: int):
        try:
            label_lines = [
                f"{detection.class_name.upper()}"
                # f"{detection.camera_id}",
                # f"{time.strftime('%H:%M:%S', time.localtime(detection.detection_time))}"
            ]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            line_height = 12
            
            max_width = 0
            for line in label_lines:
                (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                max_width = max(max_width, text_w)
            
            label_width = max_width + 8
            label_height = len(label_lines) * line_height + 6
            
            label_x = min(map_x + 15, map_image.shape[1] - label_width)
            label_y = max(map_y - label_height, 0)
            
            for i, line in enumerate(label_lines):
                text_y = label_y + 20
                cv2.putText(map_image, line, (label_x + 4, text_y), 
                           font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
                           
        except Exception as e:
            print(f"绘制透明标签失败: {e}")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        active_detections = self.get_all_active_detections()
        
        stats = {
            "total_active_detections": len(active_detections),
            "fire_count": len([d for d in active_detections if d.class_id == 0]),
            "smoke_count": len([d for d in active_detections if d.class_id == 1]),
            "cameras_with_detections": len(set(d.camera_id for d in active_detections)),
            "total_historical_detections": len(self.detection_history),
            "detection_by_camera": {}
        }
        
        for camera_id, detections in self.active_detections.items():
            valid_detections = [
                d for d in detections 
                if time.time() - d.detection_time < self.detection_timeout
            ]
            stats["detection_by_camera"][camera_id] = {
                "total": len(valid_detections),
                "fire": len([d for d in valid_detections if d.class_id == 0]),
                "smoke": len([d for d in valid_detections if d.class_id == 1])
            }
        
        return stats
    
    def cleanup_old_detections(self, max_age: float = None):
        if max_age is None:
            max_age = self.detection_timeout
        
        current_time = time.time()
        
        for camera_id in list(self.active_detections.keys()):
            self.active_detections[camera_id] = [
                detection for detection in self.active_detections[camera_id]
                if current_time - detection.detection_time < max_age
            ]
            
            if not self.active_detections[camera_id]:
                del self.active_detections[camera_id]
        
        self.detection_history = [
            detection for detection in self.detection_history
            if current_time - detection.detection_time < max_age * 2
        ]
    
    def has_active_fire_detection(self) -> bool:
        active_detections = self.get_all_active_detections()
        return len(active_detections) > 0
    
    def get_latest_detection_time(self) -> Optional[float]:
        active_detections = self.get_all_active_detections()
        if not active_detections:
            return None
        
        return max(detection.detection_time for detection in active_detections)


if __name__ == "__main__":
    detector = TunnelFireSmokeDetector()
    
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.putText(test_frame, "Fire Detection Test", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    detections = detector.detect_fire_smoke(test_frame, "cam2")
    print(f"检测结果数量: {len(detections)}")
    
    result_frame = detector.draw_detections_on_frame(test_frame, detections)
    
    test_map = np.ones((config.MAP_HEIGHT, config.MAP_WIDTH, 3), dtype=np.uint8) * 255
    result_map = detector.draw_detections_on_map(test_map)
    
    stats = detector.get_detection_statistics()
    print("检测统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("火灾检测模块测试完成")