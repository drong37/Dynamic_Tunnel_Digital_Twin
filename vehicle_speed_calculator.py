
import numpy as np
import cv2
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import traceback
import config

logger = logging.getLogger("VehicleSpeedCalculator")


@dataclass
class VehicleMotionData:
    timestamp: float
    real_position: Tuple[float, float]
    speed: float = 0.0


@dataclass
class DetectionZoneStatus:
    in_detection: bool = False
    entry_time: Optional[float] = None
    entry_pos: Optional[Tuple[int, int]] = None
    camera_id: Optional[str] = None


@dataclass
class SpeedStatistics:
    frame_count: int = 0
    speed_calculations: int = 0
    detection_speed_calculations: int = 0
    avg_calc_time: float = 0.0
    total_calc_time: float = 0.0
    max_speed: float = 0.0
    min_speed: float = float('inf')
    speed_distribution: Dict[int, int] = field(default_factory=lambda: defaultdict(int))


class MotionTracker:
    
    def __init__(self, max_history_length: int = 20):
        self.max_history_length = max_history_length
        self.history: deque = deque(maxlen=max_history_length)
    
    def add_position(self, timestamp: float, real_position: Tuple[float, float]):
        motion_data = VehicleMotionData(timestamp, real_position)
        self.history.append(motion_data)
    
    def calculate_speed(self, smooth_factor: float = 0.7) -> float:
        if len(self.history) < 2:
            return 0.0
        
        max_points = min(5, len(self.history))
        recent_history = list(self.history)[-max_points:]
        
        oldest_data = recent_history[0]
        newest_data = recent_history[-1]
        
        time_diff = newest_data.timestamp - oldest_data.timestamp
        if time_diff < 0.1:
            return recent_history[-2].speed if len(recent_history) > 1 else 0.0
        
        distance = self._calculate_distance(oldest_data.real_position, newest_data.real_position)
        speed_mps = distance / time_diff
        speed_kmh = speed_mps * 3.6
        
        if len(self.history) > 2:
            prev_speed = recent_history[-2].speed
            speed_kmh = prev_speed * smooth_factor + speed_kmh * (1 - smooth_factor)
        
        max_reasonable_speed = 80.0
        if speed_kmh > max_reasonable_speed:
            logger.warning(f"计算速度异常: {speed_kmh:.1f} km/h，限制为 {max_reasonable_speed} km/h")
            speed_kmh = max_reasonable_speed
        
        self.history[-1] = VehicleMotionData(
            newest_data.timestamp, 
            newest_data.real_position, 
            speed_kmh
        )
        
        return speed_kmh
    
    @staticmethod
    def _calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


class DetectionZoneAnalyzer:
    
    def __init__(self):
        self.detection_boundaries = self._initialize_boundaries()
        self.vehicle_status: Dict[int, DetectionZoneStatus] = {}
    
    def _initialize_boundaries(self) -> Dict[str, Tuple[int, int]]:
        boundaries = {}
        for camera_id in config.CAMERA_CONFIG.keys():
            boundary_result = config.get_camera_detection_boundaries(camera_id)
            if boundary_result[0] is not None and boundary_result[1] is not None:
                boundaries[camera_id] = boundary_result
                logger.info(f"摄像头 {camera_id} 检测区域边界: {boundary_result[0]} - {boundary_result[1]} 地图像素")
            else:
                logger.warning(f"摄像头 {camera_id} 无法获取检测区域边界")
        return boundaries
    
    def check_zone_passage(self, global_id: int, pixel_position: Tuple[int, int], 
                          timestamp: float, camera_id: str) -> Optional[float]:
        if camera_id not in self.detection_boundaries or camera_id == 'prediction':
            return None
        
        entry_x, exit_x = self.detection_boundaries[camera_id]
        current_x = pixel_position[0]
        
        if global_id not in self.vehicle_status:
            self.vehicle_status[global_id] = DetectionZoneStatus()
        
        status = self.vehicle_status[global_id]
        in_detection_zone = entry_x <= current_x <= exit_x
        
        if in_detection_zone and not status.in_detection:
            self._on_enter_detection_zone(status, pixel_position, timestamp, camera_id, global_id)
            
        elif not in_detection_zone and status.in_detection and status.camera_id == camera_id:
            speed = self._on_exit_detection_zone(status, pixel_position, timestamp, camera_id, global_id)
            return speed
        
        return None
    
    def _on_enter_detection_zone(self, status: DetectionZoneStatus, pixel_position: Tuple[int, int],
                                timestamp: float, camera_id: str, global_id: int):
        status.in_detection = True
        status.entry_time = timestamp
        status.entry_pos = pixel_position
        status.camera_id = camera_id
        logger.info(f"车辆 {global_id} 进入摄像头 {camera_id} 检测区域，位置: {pixel_position}")
    
    def _on_exit_detection_zone(self, status: DetectionZoneStatus, pixel_position: Tuple[int, int],
                               timestamp: float, camera_id: str, global_id: int) -> Optional[float]:
        if not status.entry_time or not status.entry_pos:
            self._reset_status(status)
            return None
        
        time_diff = timestamp - status.entry_time
        if time_diff <= 0.1:
            self._reset_status(status)
            return None
        
        entry_real_pos = config.pixel_to_real_coordinates(status.entry_pos[0], status.entry_pos[1])
        exit_real_pos = config.pixel_to_real_coordinates(pixel_position[0], pixel_position[1])
        
        real_distance = MotionTracker._calculate_distance(entry_real_pos, exit_real_pos)
        speed_mps = real_distance / time_diff
        speed_kmh = speed_mps * 3.6
        
        if 5 <= speed_kmh <= 150:
            self._log_detection_speed_calculation(global_id, camera_id, real_distance, time_diff, speed_kmh)
            self._reset_status(status)
            return speed_kmh
        else:
            logger.warning(f"车辆 {global_id} 通过检测区域计算速度异常: {speed_kmh:.1f} km/h，忽略")
            self._reset_status(status)
            return None
    
    def _reset_status(self, status: DetectionZoneStatus):
        status.in_detection = False
        status.entry_time = None
        status.entry_pos = None
        status.camera_id = None
    
    def _log_detection_speed_calculation(self, global_id: int, camera_id: str, 
                                       real_distance: float, time_diff: float, speed_kmh: float):
        entry_x, exit_x = self.detection_boundaries[camera_id]
        detection_length_pixels = exit_x - entry_x
        detection_length_meters = detection_length_pixels * config.COORDINATE_CALIBRATION['meters_per_pixel_x']
        
        logger.info(f"车辆 {global_id} 通过检测区域计算:")
        logger.info(f"  检测区域长度: {detection_length_meters:.2f}m ({detection_length_pixels}px)")
        logger.info(f"  实际通过距离: {real_distance:.2f}m")
        logger.info(f"  通过时间: {time_diff:.2f}s")
        logger.info(f"  计算恒定速度: {speed_kmh:.1f} km/h")
    
    def cleanup_old_status(self, current_time: float, max_age: float = 10.0):
        expired_ids = []
        for global_id, status in self.vehicle_status.items():
            if (status.entry_time is not None and 
                current_time - status.entry_time > max_age):
                expired_ids.append(global_id)
        
        for global_id in expired_ids:
            del self.vehicle_status[global_id]
            logger.debug(f"清理车辆 {global_id} 的过期检测状态")


class SpeedVisualizer:
    
    @staticmethod
    def draw_speeds_on_map(map_image: np.ndarray, vehicle_positions_map: Dict, 
                          speeds: Dict, uniform_speeds: Dict, detection_boundaries: Dict) -> np.ndarray:
        annotated_map = map_image.copy()
        
        SpeedVisualizer._draw_detection_boundaries(annotated_map, detection_boundaries)
        
        SpeedVisualizer._draw_vehicle_speeds(annotated_map, vehicle_positions_map, speeds, uniform_speeds)
        
        return annotated_map
    
    @staticmethod
    def _draw_detection_boundaries(map_image: np.ndarray, detection_boundaries: Dict):
        for camera_id, (entry_x, exit_x) in detection_boundaries.items():
            cv2.line(map_image, (entry_x, 0), (entry_x, config.MAP_HEIGHT), (0, 0, 255), 3)
            cv2.line(map_image, (exit_x, 0), (exit_x, config.MAP_HEIGHT), (0, 0, 255), 3)
            
            cv2.putText(map_image, f"{camera_id} DETECTION", 
                       (entry_x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            detection_length_pixels = exit_x - entry_x
            detection_length_meters = detection_length_pixels * config.COORDINATE_CALIBRATION['meters_per_pixel_x']
            cv2.putText(map_image, f"{detection_length_meters:.1f}m", 
                       (entry_x + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    @staticmethod
    def _draw_vehicle_speeds(map_image: np.ndarray, vehicle_positions_map: Dict, 
                           speeds: Dict, uniform_speeds: Dict):
        for global_id, vehicle_info in vehicle_positions_map.items():
            if global_id not in speeds or speeds[global_id] <= 0:
                continue
            
            pixel_position = vehicle_info['position']
            speed = speeds[global_id]
            is_uniform = global_id in uniform_speeds
            
            if is_uniform:
                color = (255, 0, 0)
                speed_text = f"{speed:.1f} km/h (恒定)"
            else:
                color = SpeedVisualizer._get_speed_color(speed)
                speed_text = f"{speed:.1f} km/h"
            
            SpeedVisualizer._draw_speed_label(map_image, pixel_position, speed_text, color)
    
    @staticmethod
    def _get_speed_color(speed: float) -> Tuple[int, int, int]:
        if speed < 30:
            return (0, 255, 0)
        elif speed < 60:
            return (0, 255, 255)
        else:
            return (0, 0, 255)
    
    @staticmethod
    def _draw_speed_label(map_image: np.ndarray, position: Tuple[int, int], 
                         text: str, color: Tuple[int, int, int]):
        text_pos = (position[0] + 5, position[1] - 10)
        
        text_pos = (
            max(0, min(text_pos[0], map_image.shape[1] - 150)),
            max(15, min(text_pos[1], map_image.shape[0] - 5))
        )
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(map_image, 
                    (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                    (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                    (0, 0, 0), -1)
        
        cv2.putText(map_image, text, text_pos, 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)


class VehicleSpeedCalculator:
    
    def __init__(self, smooth_factor: float = 0.7):
        self.smooth_factor = smooth_factor
        
        self.motion_trackers: Dict[int, MotionTracker] = {}
        self.detection_analyzer = DetectionZoneAnalyzer()
        self.current_speeds: Dict[int, float] = {}
        self.uniform_speeds: Dict[int, float] = {}
        
        self.statistics = SpeedStatistics()
        
        logger.info(f"速度计算器初始化成功，平滑因子: {smooth_factor}")
    
    def update_vehicle_positions_from_map(self, vehicle_positions_map: Dict) -> Dict[int, float]:
        try:
            start_time = time.time()
            self.statistics.frame_count += 1
            speeds = {}
            
            for global_id, vehicle_info in vehicle_positions_map.items():
                try:
                    speed = self._process_single_vehicle(global_id, vehicle_info)
                    speeds[global_id] = speed
                    self.current_speeds[global_id] = speed
                    
                    self._update_speed_statistics(speed)
                    
                except Exception as e:
                    logger.error(f"车辆ID {global_id} 速度计算错误: {e}")
                    speeds[global_id] = 0.0
            
            self._update_timing_statistics(start_time, len(vehicle_positions_map))
            
            self._cleanup_old_data()
            
            return speeds
            
        except Exception as e:
            logger.error(f"速度更新过程中发生错误: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    def _process_single_vehicle(self, global_id: int, vehicle_info: Dict) -> float:
        pixel_position = vehicle_info['position']
        timestamp = vehicle_info['timestamp']
        camera_id = vehicle_info.get('camera_id', 'unknown')
        
        real_position = config.pixel_to_real_coordinates(pixel_position[0], pixel_position[1])
        
        detection_speed = self.detection_analyzer.check_zone_passage(
            global_id, pixel_position, timestamp, camera_id
        )
        
        regular_speed = self._calculate_regular_speed(global_id, real_position, timestamp)
        
        return self._determine_final_speed(global_id, detection_speed, regular_speed)
    
    def _calculate_regular_speed(self, global_id: int, real_position: Tuple[float, float], 
                               timestamp: float) -> float:
        if global_id not in self.motion_trackers:
            self.motion_trackers[global_id] = MotionTracker()
        
        tracker = self.motion_trackers[global_id]
        tracker.add_position(timestamp, real_position)
        return tracker.calculate_speed(self.smooth_factor)
    
    def _determine_final_speed(self, global_id: int, detection_speed: Optional[float], 
                             regular_speed: float) -> float:
        if detection_speed is not None:
            self.uniform_speeds[global_id] = detection_speed
            self.statistics.detection_speed_calculations += 1
            logger.info(f"车辆 {global_id} 通过检测区域，计算恒定速度: {detection_speed:.1f} km/h")
            return detection_speed
        elif global_id in self.uniform_speeds:
            return self.uniform_speeds[global_id]
        else:
            return regular_speed
    
    def _update_speed_statistics(self, speed: float):
        if speed > 0:
            speed_bin = int(speed / 5) * 5
            self.statistics.speed_distribution[speed_bin] += 1
            
            if speed > self.statistics.max_speed:
                self.statistics.max_speed = speed
            if speed < self.statistics.min_speed:
                self.statistics.min_speed = speed
    
    def _update_timing_statistics(self, start_time: float, vehicle_count: int):
        calc_time = time.time() - start_time
        self.statistics.total_calc_time += calc_time
        self.statistics.speed_calculations += vehicle_count
        
        if self.statistics.speed_calculations > 0:
            self.statistics.avg_calc_time = (
                self.statistics.total_calc_time / self.statistics.speed_calculations
            )
    
    def _cleanup_old_data(self, max_age: float = 5.0):
        current_time = time.time()
        
        expired_trackers = []
        for global_id, tracker in self.motion_trackers.items():
            if (tracker.history and 
                current_time - tracker.history[-1].timestamp > max_age):
                expired_trackers.append(global_id)
        
        for global_id in expired_trackers:
            del self.motion_trackers[global_id]
            self.current_speeds.pop(global_id, None)
        
        self.detection_analyzer.cleanup_old_status(current_time, max_age * 2)
        
        self._cleanup_uniform_speeds()
    
    def _cleanup_uniform_speeds(self, max_age: float = 30.0):
        expired_ids = []
        for global_id in self.uniform_speeds.keys():
            if (global_id not in self.motion_trackers and 
                global_id not in self.current_speeds):
                expired_ids.append(global_id)
        
        for global_id in expired_ids:
            del self.uniform_speeds[global_id]
            logger.debug(f"清理车辆 {global_id} 的恒定速度记录")
    
    def get_uniform_speed(self, global_id: int) -> Optional[float]:
        return self.uniform_speeds.get(global_id)
    
    def set_uniform_speed(self, global_id: int, speed: float):
        self.uniform_speeds[global_id] = speed
        logger.debug(f"设置车辆 {global_id} 恒定通过速度: {speed:.1f} km/h")
    
    def get_current_speeds(self) -> Dict[int, float]:
        return self.current_speeds.copy()
    
    def get_uniform_speeds(self) -> Dict[int, float]:
        return self.uniform_speeds.copy()
    
    def draw_speeds_on_map(self, map_image: np.ndarray, vehicle_positions_map: Dict, 
                          speeds: Dict) -> np.ndarray:
        return SpeedVisualizer.draw_speeds_on_map(
            map_image, vehicle_positions_map, speeds, 
            self.uniform_speeds, self.detection_analyzer.detection_boundaries
        )
    
    def get_debug_info(self) -> Dict:
        debug_info = {
            'frame_count': self.statistics.frame_count,
            'speed_calculations': self.statistics.speed_calculations,
            'detection_speed_calculations': self.statistics.detection_speed_calculations,
            'avg_calc_time': self.statistics.avg_calc_time,
            'total_calc_time': self.statistics.total_calc_time,
            'max_speed': self.statistics.max_speed,
            'min_speed': self.statistics.min_speed if self.statistics.min_speed != float('inf') else 0,
            'speed_distribution': dict(self.statistics.speed_distribution),
            'uniform_speed_vehicles': len(self.uniform_speeds),
            'in_detection_vehicles': len([s for s in self.detection_analyzer.vehicle_status.values() if s.in_detection]),
            'detection_regions_count': len(self.detection_analyzer.detection_boundaries),
            'detection_regions_info': {}
        }
        
        for cam_id, (entry_x, exit_x) in self.detection_analyzer.detection_boundaries.items():
            length_pixels = exit_x - entry_x
            length_meters = length_pixels * config.COORDINATE_CALIBRATION['meters_per_pixel_x']
            debug_info['detection_regions_info'][cam_id] = {
                'length_pixels': length_pixels,
                'length_meters': length_meters,
                'boundaries': (entry_x, exit_x)
            }
        
        return debug_info
    
    def reset_stats(self):
        self.statistics = SpeedStatistics()
    
    def set_smooth_factor(self, smooth_factor: float):
        self.smooth_factor = max(0, min(1, smooth_factor))
        logger.info(f"更新平滑因子: {self.smooth_factor}")


if __name__ == "__main__":
    speed_calculator = VehicleSpeedCalculator(smooth_factor=0.7)
    
    print("=== 检测区域速度检测信息 ===")
    for cam_id in config.CAMERA_CONFIG.keys():
        boundaries = config.get_camera_detection_boundaries(cam_id)
        if boundaries[0] is not None:
            length_pixels = boundaries[1] - boundaries[0]
            length_meters = length_pixels * config.COORDINATE_CALIBRATION['meters_per_pixel_x']
            print(f"{cam_id}: {boundaries[0]}-{boundaries[1]} px ({length_meters:.1f}m)")
    
    test_positions = {
        1001: {'position': (100, 50), 'timestamp': 0.0, 'camera_id': 'cam1'},
        1002: {'position': (200, 75), 'timestamp': 0.0, 'camera_id': 'cam1'},
    }
    
    cam1_boundaries = config.get_camera_detection_boundaries('cam1')
    print(f"Cam1 检测区域边界: {cam1_boundaries}")
    
    for t in range(20):
        timestamp = t * 0.1
        
        if cam1_boundaries[0] is not None:
            start_x = cam1_boundaries[0] - 50
            vehicle1_x = start_x + t * 10
            vehicle2_x = start_x + t * 15
            
            test_positions[1001]['position'] = (vehicle1_x, 50)
            test_positions[1001]['timestamp'] = timestamp
            test_positions[1002]['position'] = (vehicle2_x, 75)
            test_positions[1002]['timestamp'] = timestamp
        
        speeds = speed_calculator.update_vehicle_positions_from_map(test_positions)
        print(f"时间 {timestamp:.1f}s: 车辆速度 = {speeds}")
        
        uniform_speeds = speed_calculator.get_uniform_speeds()
        if uniform_speeds:
            print(f"  恒定速度: {uniform_speeds}")
    
    debug_info = speed_calculator.get_debug_info()
    print("\n统计信息:")
    for key, value in debug_info.items():
        if key not in ['speed_distribution', 'detection_regions_info']:
            print(f"{key}: {value}")