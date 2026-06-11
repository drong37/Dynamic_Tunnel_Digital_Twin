
import os
import json
import yaml
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class CoordinateCalibration:
    pixels_per_meter_x: float
    meters_per_pixel_x: float
    pixels_per_meter_y: float
    meters_per_pixel_y: float
    origin_real_x: float = 0.0
    origin_real_y: float = 0.0


@dataclass
class TunnelDimensions:
    length: float
    width: float
    lane_width: float = 3.5


@dataclass
class CameraInfo:
    id: str
    position: str
    coverage_length: float
    video_filename: str
    calibration_points: Union[List[List[float]], np.ndarray]
    detection_region: List[Union[List[int], Tuple[int, int]]]
    lane_definitions: List[Dict[str, Any]]
    
    def __post_init__(self):
        if isinstance(self.calibration_points, list) and len(self.calibration_points) == 4:
            self.calibration_points = np.array(self.calibration_points, dtype=np.float32)
        if not isinstance(self.detection_region, list):
            self.detection_region = []


@dataclass 
class BlindZoneInfo:
    camera1: str
    camera2: str
    length: float


@dataclass
class DynamicCameraConfig:
    cameras: List[CameraInfo] = field(default_factory=list)
    blind_zones: List[BlindZoneInfo] = field(default_factory=list)
    tunnel_width: float = 7.0
    lane_width: float = 3.5
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DynamicCameraConfig':
        cameras = [CameraInfo(**cam_data) for cam_data in data.get('cameras', [])]
        blind_zones = [BlindZoneInfo(**zone_data) for zone_data in data.get('blind_zones', [])]
        
        return cls(
            cameras=cameras,
            blind_zones=blind_zones,
            tunnel_width=data.get('tunnel_width', 7.0),
            lane_width=data.get('lane_width', 3.5)
        )
    
    def get_camera_by_id(self, camera_id: str) -> Optional[CameraInfo]:
        return next((cam for cam in self.cameras if cam.id == camera_id), None)
    
    def get_camera_ids(self) -> List[str]:
        return [cam.id for cam in self.cameras]
    
    def get_camera_order(self) -> List[str]:
        return [cam.id for cam in self.cameras]


class PathConfig:
    
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models')
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.OUTPUT_DIR = os.path.join(self.DATA_DIR, 'output')
        self.INPUT_VIDEOS_DIR = os.path.join(self.DATA_DIR, 'input_videos')
        self.CONFIG_DIR = os.path.join(self.BASE_DIR, 'configs')
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        dirs = [
            self.MODEL_DIR,
            self.CONFIG_DIR,
            os.path.join(self.OUTPUT_DIR, 'processed_videos'),
            os.path.join(self.OUTPUT_DIR, 'map_visualizations')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @property
    def yolo_model_path(self) -> str:
        return os.path.join(self.MODEL_DIR, 'yolo11m.pt')
    
    @property
    def strongsort_config_path(self) -> str:
        return os.path.join(self.BASE_DIR, "strong_sort/configs/strong_sort.yaml")
    
    @property
    def strongsort_weights(self) -> str:
        return os.path.join(self.MODEL_DIR, "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pt")
    
    @property
    def camera_config_path(self) -> str:
        return os.path.join(self.CONFIG_DIR, 'camera_config.json')


class DynamicCameraConfigManager:
    
    def __init__(self, config_path: Optional[str] = None, input_videos_dir: Optional[str] = None):
        self.config_path = config_path
        self.input_videos_dir = input_videos_dir or os.path.join(os.path.dirname(__file__), 'data', 'input_videos')
        
        self.camera_config = self._load_camera_config()
        
    def _load_camera_config(self) -> DynamicCameraConfig:
        if self.config_path and os.path.exists(self.config_path):
            try:
                return self._load_from_file(self.config_path)
            except Exception as e:
                print(f"警告: 无法加载配置文件 {self.config_path}: {e}")
        
        default_paths = [
            os.path.join(os.path.dirname(__file__), 'configs', 'camera_config.json'),
            os.path.join(os.path.dirname(__file__), 'configs', 'camera_config.yaml'),
            os.path.join(os.path.dirname(__file__), 'camera_config.json'),
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                try:
                    return self._load_from_file(path)
                except Exception as e:
                    print(f"警告: 无法加载配置文件 {path}: {e}")
    
    def _load_from_file(self, file_path: str) -> DynamicCameraConfig:
        path_obj = Path(file_path)
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            if path_obj.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return DynamicCameraConfig.from_dict(data)
    
    
    def calculate_tunnel_dimensions(self) -> TunnelDimensions:
        total_coverage = sum(cam.coverage_length for cam in self.camera_config.cameras)
        total_blind_zones = sum(zone.length for zone in self.camera_config.blind_zones)
        total_length = total_coverage + total_blind_zones
        
        return TunnelDimensions(
            length=total_length,
            width=self.camera_config.tunnel_width,
            lane_width=self.camera_config.lane_width
        )
    
    def calculate_coordinate_calibration(self, tunnel_dims: TunnelDimensions, 
                                       map_width: int, map_height: int) -> CoordinateCalibration:
        return CoordinateCalibration(
            pixels_per_meter_x=map_width / tunnel_dims.length,
            meters_per_pixel_x=tunnel_dims.length / map_width,
            pixels_per_meter_y=map_height / tunnel_dims.width,
            meters_per_pixel_y=tunnel_dims.width / map_height
        )
    
    def calculate_camera_map_regions(self, calibration: CoordinateCalibration, map_height: int) -> Dict[str, Dict]:
        regions = {}
        current_x = 0
        pixels_per_meter = calibration.pixels_per_meter_x
        
        camera_order = self.camera_config.get_camera_order()
        
        for i, cam_id in enumerate(camera_order):
            camera_info = self.camera_config.get_camera_by_id(cam_id)
            if camera_info is None:
                continue
            coverage_length = camera_info.coverage_length
            pixel_length = int(coverage_length * pixels_per_meter)
            
            x_min = current_x
            x_max = current_x + pixel_length
            
            regions[cam_id] = {
                'map_region': (x_min, 0, x_max, map_height),
                'real_start': current_x / pixels_per_meter,
                'real_end': x_max / pixels_per_meter,
                'coverage_length': coverage_length
            }
            
            current_x = x_max
            
            if i < len(camera_order) - 1:
                next_cam = camera_order[i + 1]
                blind_zone = self._find_blind_zone(cam_id, next_cam)
                if blind_zone:
                    blind_zone_pixels = int(blind_zone.length * pixels_per_meter)
                    current_x += blind_zone_pixels
        
        return regions
    
    def _find_blind_zone(self, cam1: str, cam2: str) -> Optional[BlindZoneInfo]:
        for zone in self.camera_config.blind_zones:
            if (zone.camera1 == cam1 and zone.camera2 == cam2) or \
               (zone.camera1 == cam2 and zone.camera2 == cam1):
                return zone
        return None
    
    def generate_camera_config(self, map_regions: Dict, map_height: int) -> Dict[str, Dict]:
        config = {}
        
        for camera_info in self.camera_config.cameras:
            cam_id = camera_info.id
            map_region = map_regions[cam_id]['map_region']
            src_points = np.array(camera_info.calibration_points, dtype=np.float32)
            
            dst_points = np.array([
                [map_region[0], 0],
                [map_region[0], map_height],
                [map_region[2], 0],
                [map_region[2], map_height]
            ], dtype=np.float32)
            
            config[cam_id] = {
                'id': cam_id,
                'position': camera_info.position,
                'video_path': camera_info.video_filename if camera_info.video_filename.startswith(('rtsp://', 'http://', 'https://', 'rtmp://')) else os.path.join(self.input_videos_dir, camera_info.video_filename),
                'coverage_length': camera_info.coverage_length,
                'calibration': {
                    'map_region': map_region,
                    'src_points': src_points,
                    'dst_points': dst_points
                }
            }
        
        return config
    
    def generate_detection_regions(self) -> Dict[str, List]:
        return {cam.id: cam.detection_region for cam in self.camera_config.cameras}
    
    def generate_lane_definitions(self) -> Dict[str, List]:
        return {cam.id: cam.lane_definitions for cam in self.camera_config.cameras}
    
    def generate_camera_topology(self) -> Dict[str, List[str]]:
        topology = {}
        camera_order = self.camera_config.get_camera_order()
        
        for i, cam_id in enumerate(camera_order):
            neighbors = []
            
            # if i > 0:
            #     neighbors.append(camera_order[i - 1])
            
            if i < len(camera_order) - 1:
                neighbors.append(camera_order[i + 1])
            
            topology[cam_id] = neighbors
        
        return topology
    
    def generate_blind_zones_dict(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        blind_zones = {}
        
        for zone in self.camera_config.blind_zones:
            key = (zone.camera1, zone.camera2)
            
            blind_zones[key] = {
                'length': zone.length,
                'start_x': 0,
                'end_x': 0
            }
        
        return blind_zones


class CoordinateTransformer:
    
    def __init__(self, calibration: CoordinateCalibration, map_height: int):
        self.calibration = calibration
        self.map_height = map_height
    
    def pixel_to_real(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        real_x = self.calibration.origin_real_x + pixel_x * self.calibration.meters_per_pixel_x
        real_y = self.calibration.origin_real_y + (self.map_height - pixel_y) * self.calibration.meters_per_pixel_y
        return real_x, real_y
    
    def real_to_pixel(self, real_x: float, real_y: float) -> Tuple[int, int]:
        pixel_x = (real_x - self.calibration.origin_real_x) * self.calibration.pixels_per_meter_x
        pixel_y = self.map_height - (real_y - self.calibration.origin_real_y) * self.calibration.pixels_per_meter_y
        return int(pixel_x), int(pixel_y)
    
    def pixel_distance_to_real(self, pixel_distance_x: float, pixel_distance_y: float) -> Tuple[float, float]:
        real_distance_x = pixel_distance_x * self.calibration.meters_per_pixel_x
        real_distance_y = pixel_distance_y * self.calibration.meters_per_pixel_y
        return real_distance_x, real_distance_y


class DetectionRegionManager:
    
    def __init__(self, camera_config: Dict, detection_regions: Dict):
        self.camera_config = camera_config
        self.detection_regions = detection_regions
    
    def transform_detection_region_to_map(self, camera_id: str) -> Optional[Tuple[int, int]]:
        if camera_id not in self.detection_regions or camera_id not in self.camera_config:
            return None
        
        detection_polygon = self.detection_regions[camera_id]
        src_points = self.camera_config[camera_id]['calibration']['src_points']
        dst_points = self.camera_config[camera_id]['calibration']['dst_points']
        
        try:
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            detection_points = np.array(detection_polygon, dtype=np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(detection_points, transform_matrix)
            transformed_points = transformed_points.reshape(-1, 2)
            
            x_coords = transformed_points[:, 0]
            min_x = max(0, min(np.min(x_coords), MAP_WIDTH))
            max_x = max(0, min(np.max(x_coords), MAP_WIDTH))
            
            return int(min_x), int(max_x)
            
        except Exception as e:
            print(f"转换检测区域失败 {camera_id}: {e}")
            return self._get_fallback_boundaries(camera_id)
    
    def _get_fallback_boundaries(self, camera_id: str) -> Optional[Tuple[int, int]]:
        if camera_id not in self.camera_config:
            return None
        
        region = self.camera_config[camera_id]['calibration']['map_region']
        x_min, _, x_max, _ = region
        
        detection_width = (x_max - x_min) * 0.8
        detection_start = x_min + (x_max - x_min) * 0.1
        detection_end = detection_start + detection_width
        
        return int(detection_start), int(detection_end)


# =============================================================================
# =============================================================================

paths = PathConfig()

config_file_path = os.environ.get('CAMERA_CONFIG_PATH', None)
camera_manager = DynamicCameraConfigManager(config_file_path, paths.INPUT_VIDEOS_DIR)

MAP_WIDTH = 1500
MAP_HEIGHT = 100
MAP_BG_COLOR = (255, 255, 255)

tunnel_dimensions = camera_manager.calculate_tunnel_dimensions()
coordinate_calibration = camera_manager.calculate_coordinate_calibration(tunnel_dimensions, MAP_WIDTH, MAP_HEIGHT)
camera_map_regions = camera_manager.calculate_camera_map_regions(coordinate_calibration, MAP_HEIGHT)
camera_config = camera_manager.generate_camera_config(camera_map_regions, MAP_HEIGHT)

coordinate_transformer = CoordinateTransformer(coordinate_calibration, MAP_HEIGHT)
detection_manager = DetectionRegionManager(camera_config, camera_manager.generate_detection_regions())

def calculate_blind_zones() -> Dict:
    blind_zones = {}
    
    camera_order = camera_manager.camera_config.get_camera_order()
    
    for i in range(len(camera_order) - 1):
        cam1 = camera_order[i]
        cam2 = camera_order[i + 1]
        
        cam1_end = camera_map_regions[cam1]['map_region'][2]
        cam2_start = camera_map_regions[cam2]['map_region'][0]
        
        if cam2_start > cam1_end:
            blind_zones[(cam1, cam2)] = {
                'start_x': cam1_end, 
                'end_x': cam2_start
            }
    
    return blind_zones

# =============================================================================
# =============================================================================

BASE_DIR = paths.BASE_DIR
MODEL_DIR = paths.MODEL_DIR
OUTPUT_DIR = paths.OUTPUT_DIR
INPUT_VIDEOS_DIR = paths.INPUT_VIDEOS_DIR
YOLO_MODEL_PATH = paths.yolo_model_path
STRONGSORT_CONFIG_PATH = paths.strongsort_config_path
STRONGSORT_WEIGHTS = paths.strongsort_weights

TUNNEL_REAL_DIMENSIONS = {
    'length': tunnel_dimensions.length,
    'width': tunnel_dimensions.width,
    'lane_width': tunnel_dimensions.lane_width
}

COORDINATE_CALIBRATION = {
    'pixels_per_meter_x': coordinate_calibration.pixels_per_meter_x,
    'meters_per_pixel_x': coordinate_calibration.meters_per_pixel_x,
    'pixels_per_meter_y': coordinate_calibration.pixels_per_meter_y,
    'meters_per_pixel_y': coordinate_calibration.meters_per_pixel_y,
    'origin_real_x': coordinate_calibration.origin_real_x,
    'origin_real_y': coordinate_calibration.origin_real_y,
}

CAMERA_CONFIG = camera_config
CAMERA_MAP_REGIONS = camera_map_regions
DETECTION_REGIONS = camera_manager.generate_detection_regions()
BLIND_ZONES = calculate_blind_zones()
LANE_DEFINITIONS = camera_manager.generate_lane_definitions()
CAMERA_TOPOLOGY = camera_manager.generate_camera_topology()

def pixel_to_real_coordinates(pixel_x: float, pixel_y: float) -> Tuple[float, float]:
    return coordinate_transformer.pixel_to_real(pixel_x, pixel_y)

def real_to_pixel_coordinates(real_x: float, real_y: float) -> Tuple[int, int]:
    return coordinate_transformer.real_to_pixel(real_x, real_y)

def pixel_distance_to_real_distance(pixel_distance_x: float, pixel_distance_y: float) -> Tuple[float, float]:
    return coordinate_transformer.pixel_distance_to_real(pixel_distance_x, pixel_distance_y)

def get_camera_detection_boundaries(camera_id: str) -> Tuple[Optional[int], Optional[int]]:
    boundaries = detection_manager.transform_detection_region_to_map(camera_id)
    if boundaries:
        return boundaries
    return None, None

CONFIDENCE_THRESHOLD = 0.3
VEHICLE_CLASSES = [2, 5, 7]
REID_MATCH_THRESHOLD = 0.65
TARGET_FPS = 7

TRACKING_STRATEGY: Dict[str, Any] = {
    'time_windows': {
        'prediction_start_delay_sec': 0.5,
        'prediction_timeout_sec': 60.0,
        'lost_vehicle_timeout_sec': 60.0,
        'max_prediction_time_sec': 60.0,
        'prediction_update_interval_sec': 0.01,

        'trajectory_timeout_sec': 30.0,
        'ghost_check_interval_sec': 3.0,
        'ghost_no_detection_sec': 10.0,
        'ghost_suspect_window_sec': 2.0,
        'ghost_quarantine_window_sec': 6.0,
        'ghost_event_log_limit': 2000,
        'blind_zone_protection_enabled': True,
    },
    'active_tracking': {
        'mapping_recent_sec': 5.0,
        'detection_recent_sec': 3.0,
        'detection_stale_sec': 5.0,
        'camera_recent_sec': 3.0,
    },
    'suppression': {
        'base_pixels': 5.0,
        'min_pixels': 5,
        'max_pixels': 20.0,
        'time_horizon_sec': 0.30,
    },
    'ghost_scoring': {
        'enabled': True,
        'enter_score_threshold': 0.60,
        'exit_score_threshold': 0.45,
        'delete_score_threshold': 0.75,

        'lane_tolerance_pixels': 12.0,
        'inactive_floor_sec': 1.0,

        'weights': {
            'time_window': 0.35,
            'tracking_inactive': 0.25,
            'predictor_health': 0.15,
            'lane_consistency': 0.10,
            'direction_consistency': 0.10,
            'camera_region': 0.05,
        }
    }
}


def validate_tracking_strategy() -> None:
    windows = TRACKING_STRATEGY['time_windows']
    active = TRACKING_STRATEGY['active_tracking']
    suppression = TRACKING_STRATEGY['suppression']
    ghost_scoring = TRACKING_STRATEGY.get('ghost_scoring', {})

    prediction_start_delay = float(windows['prediction_start_delay_sec'])
    prediction_timeout = float(windows['prediction_timeout_sec'])
    max_prediction_time = float(windows['max_prediction_time_sec'])
    ghost_check_interval = float(windows['ghost_check_interval_sec'])
    ghost_suspect_window = float(windows['ghost_suspect_window_sec'])
    ghost_quarantine_window = float(windows['ghost_quarantine_window_sec'])
    detection_recent = float(active['detection_recent_sec'])
    detection_stale = float(active['detection_stale_sec'])
    min_pixels = float(suppression['min_pixels'])
    base_pixels = float(suppression['base_pixels'])
    max_pixels = float(suppression['max_pixels'])
    enter_score = float(ghost_scoring.get('enter_score_threshold', 0.60))
    exit_score = float(ghost_scoring.get('exit_score_threshold', 0.45))
    delete_score = float(ghost_scoring.get('delete_score_threshold', 0.75))
    lane_tolerance = float(ghost_scoring.get('lane_tolerance_pixels', 12.0))
    inactive_floor = float(ghost_scoring.get('inactive_floor_sec', 1.0))
    weight_cfg = ghost_scoring.get('weights', {})
    weights_sum = float(sum(float(v) for v in weight_cfg.values())) if weight_cfg else 0.0

    assert prediction_start_delay >= 0.0, "prediction_start_delay_sec 必须 >= 0"
    assert max_prediction_time <= prediction_timeout, "max_prediction_time_sec 必须 <= prediction_timeout_sec"
    assert ghost_check_interval <= prediction_timeout / 2.0, "ghost_check_interval_sec 建议 <= prediction_timeout_sec/2"
    assert 0.0 <= ghost_suspect_window < ghost_quarantine_window, "ghost_suspect_window_sec 必须小于 ghost_quarantine_window_sec"
    assert ghost_quarantine_window <= prediction_timeout, "ghost_quarantine_window_sec 必须 <= prediction_timeout_sec"
    assert detection_recent <= detection_stale, "detection_recent_sec 必须 <= detection_stale_sec"
    assert min_pixels <= base_pixels <= max_pixels, "suppression 阈值必须满足 min <= base <= max"
    assert 0.0 <= exit_score <= enter_score <= delete_score <= 1.0, "ghost_scoring 阈值必须满足 0 <= exit <= enter <= delete <= 1"
    assert lane_tolerance > 0.0, "lane_tolerance_pixels 必须 > 0"
    assert 0.0 <= inactive_floor <= prediction_timeout, "inactive_floor_sec 必须在 [0, prediction_timeout_sec] 区间"
    assert 0.99 <= weights_sum <= 1.01, "ghost_scoring.weights 的总和应约等于 1.0"


validate_tracking_strategy()

MAP_LANE_DEFINITIONS = [
    {"name": "lane1", "y_range": (0, 50), "x_range": (0, MAP_WIDTH)},
    {"name": "lane2", "y_range": (50, MAP_HEIGHT), "x_range": (0, MAP_WIDTH)}
]

# =============================================================================
# =============================================================================

def get_camera_count() -> int:
    return len(camera_manager.camera_config.cameras)

def get_camera_list() -> List[str]:
    return camera_manager.camera_config.get_camera_ids()

def add_camera(camera_info: CameraInfo) -> bool:
    try:
        camera_manager.camera_config.cameras.append(camera_info)
        _recalculate_global_config()
        return True
    except Exception as e:
        print(f"添加摄像头失败: {e}")
        return False

def remove_camera(camera_id: str) -> bool:
    try:
        camera_manager.camera_config.cameras = [
            cam for cam in camera_manager.camera_config.cameras 
            if cam.id != camera_id
        ]
        _recalculate_global_config()
        return True
    except Exception as e:
        print(f"移除摄像头失败: {e}")
        return False

def _recalculate_global_config():
    global tunnel_dimensions, coordinate_calibration, camera_map_regions, camera_config
    global TUNNEL_REAL_DIMENSIONS, COORDINATE_CALIBRATION, CAMERA_CONFIG
    global CAMERA_MAP_REGIONS, DETECTION_REGIONS, BLIND_ZONES, LANE_DEFINITIONS, CAMERA_TOPOLOGY
    
    tunnel_dimensions = camera_manager.calculate_tunnel_dimensions()
    coordinate_calibration = camera_manager.calculate_coordinate_calibration(tunnel_dimensions, MAP_WIDTH, MAP_HEIGHT)
    camera_map_regions = camera_manager.calculate_camera_map_regions(coordinate_calibration, MAP_HEIGHT)
    camera_config = camera_manager.generate_camera_config(camera_map_regions, MAP_HEIGHT)
    
    TUNNEL_REAL_DIMENSIONS = {
        'length': tunnel_dimensions.length,
        'width': tunnel_dimensions.width,
        'lane_width': tunnel_dimensions.lane_width
    }
    
    COORDINATE_CALIBRATION = {
        'pixels_per_meter_x': coordinate_calibration.pixels_per_meter_x,
        'meters_per_pixel_x': coordinate_calibration.meters_per_pixel_x,
        'pixels_per_meter_y': coordinate_calibration.pixels_per_meter_y,
        'meters_per_pixel_y': coordinate_calibration.meters_per_pixel_y,
        'origin_real_x': coordinate_calibration.origin_real_x,
        'origin_real_y': coordinate_calibration.origin_real_y,
    }
    
    CAMERA_CONFIG = camera_config
    CAMERA_MAP_REGIONS = camera_map_regions
    DETECTION_REGIONS = camera_manager.generate_detection_regions()
    BLIND_ZONES = calculate_blind_zones()
    LANE_DEFINITIONS = camera_manager.generate_lane_definitions()
    CAMERA_TOPOLOGY = camera_manager.generate_camera_topology()

def save_current_config(output_path: str):
    camera_manager.camera_config.to_dict()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            yaml.dump(camera_manager.camera_config.to_dict(), f, default_flow_style=False, allow_unicode=True, indent=2)
        else:
            json.dump(camera_manager.camera_config.to_dict(), f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {output_path}")

def print_camera_config_info():
    print("=== 动态摄像头配置信息 ===")
    print(f"摄像头数量: {get_camera_count()}")
    print(f"隧道总长度: {TUNNEL_REAL_DIMENSIONS['length']:.1f}m")
    print(f"隧道总宽度: {TUNNEL_REAL_DIMENSIONS['width']:.1f}m")
    print(f"地图尺寸: {MAP_WIDTH}x{MAP_HEIGHT} 像素")
    print(f"像素比例: {COORDINATE_CALIBRATION['pixels_per_meter_x']:.2f} 像素/米")
    print()
    
    for cam_id in camera_manager.camera_config.get_camera_order():
        config_info = CAMERA_CONFIG[cam_id]
        region = config_info['calibration']['map_region']
        coverage = config_info['coverage_length']
        real_start = CAMERA_MAP_REGIONS[cam_id]['real_start']
        real_end = CAMERA_MAP_REGIONS[cam_id]['real_end']
        
        print(f"{cam_id} ({config_info['position']}):")
        print(f"  覆盖长度: {coverage}m ({real_start:.1f}m - {real_end:.1f}m)")
        print(f"  地图区域: {region} 像素")
        
        detection_boundaries = get_camera_detection_boundaries(cam_id)
        if detection_boundaries[0] is not None and detection_boundaries[1] is not None:
            detection_length_pixels = detection_boundaries[1] - detection_boundaries[0]
            detection_length_meters = detection_length_pixels * COORDINATE_CALIBRATION['meters_per_pixel_x']
            print(f"  检测区域: {detection_boundaries[0]}-{detection_boundaries[1]} 像素 ({detection_length_meters:.1f}m)")
        print()
    
    print("盲区信息:")
    for (cam1, cam2), zone in BLIND_ZONES.items():
        length_pixels = zone['end_x'] - zone['start_x']
        length_meters = length_pixels * COORDINATE_CALIBRATION['meters_per_pixel_x']
        print(f"  {cam1} -> {cam2}: {zone['start_x']}-{zone['end_x']} 像素 ({length_meters:.1f}m)")
    
    print(f"\n摄像头拓扑: {CAMERA_TOPOLOGY}")

def print_initialization_info():
    config_source = "配置文件" if camera_manager.config_path else "默认配置"
    print(f"[OK] 动态摄像头配置系统已初始化")
    print(f"[OK] 配置来源: {config_source}")
    print(f"[OK] 摄像头数量: {get_camera_count()}")
    print(f"[OK] 支持动态扩展，可通过配置文件或API添加摄像头")
    
    if not camera_manager.config_path:
        template_path = os.path.join(paths.CONFIG_DIR, 'camera_config_template.json')
        print(f"[OK] 可创建配置模板: create_config_template('{template_path}')")

WEBSOCKET_CONFIG = {
    'enabled': True,
    'host': '0.0.0.0',
    'port': 8765,

    'project_code': '200693',
    'dept_id': '1934428917679534082',

    'max_queue_size': 100,

    'tunnel_length': TUNNEL_REAL_DIMENSIONS['length'],
    'total_lanes': len(MAP_LANE_DEFINITIONS),

    'incremental_enabled': True,
    'position_change_threshold': 0.005,
    'flush_interval_ms': 20,
}

if __name__ != "__main__":
    print_initialization_info()

if __name__ == "__main__":
    print_camera_config_info()
