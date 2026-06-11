import cv2
import numpy as np
import config

class CameraCalibration:
    
    def __init__(self, camera_id):

        self.camera_id = camera_id
        
        if camera_id in config.CAMERA_CONFIG:
            cam_config = config.CAMERA_CONFIG[camera_id]
            self.map_region = cam_config['calibration']['map_region']
            self.src_points = cam_config['calibration']['src_points']
            self.dst_points = cam_config['calibration']['dst_points']
            
            if len(self.src_points) != len(self.dst_points):
                raise ValueError(f"摄像头 {camera_id} 的源点和目标点数量不一致")
                
            if self.src_points.shape[0] != 4 or self.dst_points.shape[0] != 4:
                raise ValueError(f"摄像头 {camera_id} 的标定点必须是4个")
                
            print(f"摄像头 {camera_id} 标定信息:")
            print(f"源点: {self.src_points}")
            print(f"目标点: {self.dst_points}")
            
            try:
                self.homography_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
                print(f"单应性矩阵计算成功: \n{self.homography_matrix}")
                
                test_points = np.array([[800, 800]], dtype=np.float32)
                for test_pt in test_points:
                    p = np.array([test_pt[0], test_pt[1], 1], dtype=np.float32)
                    px, py, pw = np.dot(self.homography_matrix, p)
                    mapped_x, mapped_y = px/pw, py/pw
                    print(f"测试点 {test_pt} -> 映射到 ({mapped_x:.1f}, {mapped_y:.1f})")
            except Exception as e:
                print(f"单应性矩阵计算失败: {e}")
                import traceback
                traceback.print_exc()
                raise ValueError(f"摄像头 {camera_id} 的单应性矩阵计算失败")
                
        else:
            raise ValueError(f"未知摄像头ID: {camera_id}")
    
    def map_to_ground(self, pixel_point):

        try:
            if not isinstance(pixel_point, (list, tuple, np.ndarray)) or len(pixel_point) < 2:
                print(f"警告: 无效的像素坐标: {pixel_point}")
                return [0, 0]
                
            if np.isnan(pixel_point[0]) or np.isnan(pixel_point[1]):
                print(f"警告: 像素坐标包含NaN值: {pixel_point}")
                return [0, 0]
            
            x, y = float(pixel_point[0]), float(pixel_point[1])
            
            h = self.homography_matrix
            h00, h01, h02 = h[0,0], h[0,1], h[0,2]
            h10, h11, h12 = h[1,0], h[1,1], h[1,2]
            h20, h21, h22 = h[2,0], h[2,1], h[2,2]

            x_prime = h00 * x + h01 * y + h02
            y_prime = h10 * x + h11 * y + h12
            w_prime = h20 * x + h21 * y + h22

            if abs(w_prime) > 1e-10:
                map_x = x_prime / w_prime
                map_y = y_prime / w_prime
            else:
                print(f"警告: 坐标变换失败，透视除法分母接近零 (w'={w_prime})")
                return [0, 0]
            
            map_x = np.clip(map_x, 0, config.MAP_WIDTH)
            map_y = np.clip(map_y, 0, config.MAP_HEIGHT)
            
            try:
                pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pts, self.homography_matrix)
                cv_result = transformed[0][0]
                
            except Exception as cv_error:
                print(f"OpenCV变换失败: {cv_error}")
            
            return [map_x, map_y]
            
        except Exception as e:
            print(f"坐标映射错误: {e}")
            import traceback
            traceback.print_exc()
            return [0, 0]
    
    def draw_mapping_grid(self, frame, grid_size=50):
        h, w = frame.shape[:2]
        vis_frame = frame.copy()
        
        for x in range(0, w, grid_size):
            cv2.line(vis_frame, (x, 0), (x, h), (0, 255, 0), 1)
        for y in range(0, h, grid_size):
            cv2.line(vis_frame, (0, y), (w, y), (0, 255, 0), 1)

        for i, point in enumerate(self.src_points):
            cv2.circle(vis_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            cv2.putText(vis_frame, f"P{i}", (int(point[0])+5, int(point[1])-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        bottom_points = []
        for x in range(grid_size, w, grid_size):
            bottom_y = h - grid_size
            map_point = self.map_to_ground([x, bottom_y])
            bottom_points.append(((x, bottom_y), map_point))
            cv2.circle(vis_frame, (x, bottom_y), 3, (255, 0, 0), -1)
            label = f"({int(map_point[0])},{int(map_point[1])})"
            cv2.putText(vis_frame, label, (x-30, bottom_y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return vis_frame
    
    def visualize_mapping(self):
        h, w = 480, 640
        camera_view = np.ones((h, w, 3), dtype=np.uint8) * 255

        grid_size = 50
        for x in range(0, w, grid_size):
            cv2.line(camera_view, (x, 0), (x, h), (200, 200, 200), 1)
        for y in range(0, h, grid_size):
            cv2.line(camera_view, (0, y), (w, y), (200, 200, 200), 1)

        for i, point in enumerate(self.src_points):
            if point[0] < w and point[1] < h:
                cv2.circle(camera_view, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                cv2.putText(camera_view, f"P{i}", (int(point[0])+5, int(point[1])-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        map_view = np.ones((config.MAP_HEIGHT, config.MAP_WIDTH, 3), dtype=np.uint8) * 255

        x_min, y_min, x_max, y_max = self.map_region
        cv2.rectangle(map_view, (x_min, y_min), (x_max, y_max), (200, 200, 200), 2)
        cv2.putText(map_view, f"Camera {self.camera_id} Region", (x_min+10, y_min-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
     
        for i, point in enumerate(self.dst_points):
            cv2.circle(map_view, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            cv2.putText(map_view, f"P{i}", (int(point[0])+5, int(point[1])-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for x in range(0, w, grid_size):
            for y in range(h-grid_size, h, grid_size):
                map_point = self.map_to_ground([x, y])
                mx, my = int(map_point[0]), int(map_point[1])

                if 0 <= mx < config.MAP_WIDTH and 0 <= my < config.MAP_HEIGHT:
                    cv2.circle(map_view, (mx, my), 3, (0, 255, 0), -1)

                    cv2.circle(camera_view, (x, y), 3, (0, 255, 0), -1)

        visualization = np.ones((h, w*2 + 20, 3), dtype=np.uint8) * 255
        visualization[:, :w] = camera_view

        map_view_resized = cv2.resize(map_view, (w, h))
        visualization[:, w+20:] = map_view_resized

        cv2.putText(visualization, "Camera View", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(visualization, "Map View", (w+30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return visualization

if __name__ == "__main__":
    for cam_id in config.CAMERA_CONFIG:
        calibration = CameraCalibration(cam_id)
        vis_img = calibration.visualize_mapping()
        cv2.imshow(f'Calibration Visualization - {cam_id}', vis_img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()