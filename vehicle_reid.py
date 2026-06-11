
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import os
import time
import glob
import cv2
from PIL import Image
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Union
from C2TReID.utils.reranking import re_ranking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedVehicleReID")

class EnhancedSimilarityCalculator:
    
    def __init__(self):
        self.similarity_history = deque(maxlen=1000)
        
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray, 
                           quality_factor: float = 1.0, 
                           camera_context: Optional[Dict] = None) -> float:
        if features1 is None or features2 is None:
            return 0.0
            
        try:
            cos_sim = np.dot(features1, features2)
            
            if cos_sim > 0.7:
                euclidean_dist = np.linalg.norm(features1 - features2)
                max_dist = np.sqrt(2)
                normalized_dist = euclidean_dist / max_dist
                dist_penalty = normalized_dist * 0.1
                cos_sim = max(0, cos_sim - dist_penalty)
            
            quality_factor = max(0.0, min(1.0, float(quality_factor)))
            quality_weight = 0.85 + 0.15 * quality_factor
            adjusted_sim = cos_sim * quality_weight
            
            if camera_context:
                if camera_context.get('same_camera', False):
                    adjusted_sim *= 0.93
                elif camera_context.get('adjacent_cameras', False):
                    adjusted_sim *= 1.02
                    
            self.similarity_history.append(adjusted_sim)
            
            return min(1.0, max(0.0, adjusted_sim))
            
        except Exception as e:
            logger.error(f"相似度计算错误: {e}")
            return 0.0
    
    def get_similarity_statistics(self) -> Dict:
        if not self.similarity_history:
            return {}
            
        similarities = list(self.similarity_history)
        return {
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'median': np.median(similarities),
            'count': len(similarities)
        }

class AdaptiveThresholdManager:
    
    def __init__(self, base_threshold: float = 0.6, adaptation_rate: float = 0.01):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.adaptation_rate = adaptation_rate
        self.match_history = deque(maxlen=200)
        self.false_positive_count = 0
        self.true_positive_count = 0
        
    def update_match_result(self, similarity: float, is_correct_match: bool):
        self.match_history.append((similarity, is_correct_match))
        
        if similarity > self.current_threshold:
            if is_correct_match:
                self.true_positive_count += 1
            else:
                self.false_positive_count += 1
                
        if len(self.match_history) >= 50:
            self._adjust_threshold()
            
    def _adjust_threshold(self):
        total_matches = self.true_positive_count + self.false_positive_count
        if total_matches == 0:
            return
            
        false_positive_rate = self.false_positive_count / total_matches
        
        if false_positive_rate > 0.15:
            adjustment = min(0.05, false_positive_rate * 0.2)
            self.current_threshold = min(0.9, self.current_threshold + adjustment)
            logger.debug(f"提高匹配阈值至 {self.current_threshold:.3f}")
            
        elif false_positive_rate < 0.05 and self.true_positive_count > 10:
            adjustment = min(0.03, (0.05 - false_positive_rate) * 0.1)
            self.current_threshold = max(0.4, self.current_threshold - adjustment)
            logger.debug(f"降低匹配阈值至 {self.current_threshold:.3f}")
            
        self.false_positive_count = 0
        self.true_positive_count = 0
        
    def get_threshold(self, camera_id: Optional[str] = None) -> float:
        return self.current_threshold
    
    def get_stats(self) -> Dict:
        return {
            'current_threshold': self.current_threshold,
            'base_threshold': self.base_threshold,
            'match_history_size': len(self.match_history),
            'tp_count': self.true_positive_count,
            'fp_count': self.false_positive_count
        }

class FeatureQualityAssessor:
    
    def assess_image_quality(self, vehicle_patch: np.ndarray) -> float:
        if vehicle_patch is None:
            return 0.0
            
        try:
            if len(vehicle_patch.shape) == 3:
                gray = cv2.cvtColor(vehicle_patch, cv2.COLOR_BGR2GRAY)
            else:
                gray = vehicle_patch
                
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            
            mean_brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
            brightness_score = max(0.0, brightness_score)
            
            contrast = np.std(gray) / 255.0
            contrast_score = min(1.0, contrast * 3)
            
            h, w = vehicle_patch.shape[:2]
            size_ratio = (h * w) / (100 * 100)
            size_score = min(1.0, size_ratio)
            
            overall_score = (sharpness_score * 0.4 + 
                           brightness_score * 0.2 + 
                           contrast_score * 0.2 + 
                           size_score * 0.2)
            
            return float(max(0.0, min(1.0, overall_score)))
            
        except Exception as e:
            logger.error(f"质量评估错误: {e}")
            return 0.5

class MultiFeatureManager:
    
    def __init__(self, max_features_per_id: int = 6, max_features_per_camera: int = 2):
        self.max_features_per_id = max_features_per_id
        self.max_features_per_camera = max_features_per_camera
        self.feature_records = defaultdict(list)  # gid -> [(feature, quality, timestamp), ...]
        self.camera_feature_records = defaultdict(lambda: defaultdict(list))  # gid -> {camera_id: [(feature, quality, timestamp), ...]}

    @staticmethod
    def _trim_records(records: List[Tuple[np.ndarray, float, float]], max_count: int):
        if len(records) > max_count:
            records.sort(key=lambda x: x[1], reverse=True)
            del records[max_count:]
        
    def add_feature(self, global_id: int, feature: np.ndarray, 
                   quality: float, timestamp: Optional[float] = None,
                   camera_id: Optional[str] = None):
        if timestamp is None:
            timestamp = time.time()
            
        record = (feature.copy(), quality, timestamp)
        self.feature_records[global_id].append(record)
        self._trim_records(self.feature_records[global_id], self.max_features_per_id)

        if camera_id is not None:
            bucket = self.camera_feature_records[global_id][camera_id]
            bucket.append(record)
            self._trim_records(bucket, self.max_features_per_camera)
            
    def get_best_feature(self, global_id: int) -> Optional[np.ndarray]:
        if global_id not in self.feature_records:
            return None
            
        records = self.feature_records[global_id]
        if not records:
            return None
            
        best_record = max(records, key=lambda x: x[1])
        return best_record[0]
    
    def _collect_feature_records(self, global_id: int,
                               camera_id: Optional[str] = None,
                               camera_topology: Optional[Dict[str, List[str]]] = None) -> List[Tuple[np.ndarray, float, float, str]]:
        if global_id not in self.feature_records:
            return []

        combined_records: List[Tuple[np.ndarray, float, float, str]] = []
        seen_features = set()

        def append_unique(records: List[Tuple[np.ndarray, float, float]], source: str):
            for record in records:
                feature_id = id(record[0])
                if feature_id in seen_features:
                    continue
                seen_features.add(feature_id)
                combined_records.append((record[0], record[1], record[2], source))

        camera_buckets = self.camera_feature_records.get(global_id, {})
        if camera_id is not None:
            append_unique(camera_buckets.get(camera_id, []), 'same')

            if camera_topology:
                for neighbor_camera in camera_topology.get(camera_id, []):
                    append_unique(camera_buckets.get(neighbor_camera, []), 'adjacent')

        append_unique(self.feature_records[global_id], 'global')
        return combined_records

    def get_all_features(self, global_id: int, camera_id: Optional[str] = None,
                        camera_topology: Optional[Dict[str, List[str]]] = None) -> List[np.ndarray]:
        records = self._collect_feature_records(global_id, camera_id, camera_topology)
        return [record[0] for record in records]
    
    def calculate_multi_similarity(self, query_feature: np.ndarray, 
                                 global_id: int, 
                                 similarity_calculator: EnhancedSimilarityCalculator,
                                 camera_id: Optional[str] = None,
                                 camera_topology: Optional[Dict[str, List[str]]] = None) -> float:
        records = self._collect_feature_records(global_id, camera_id, camera_topology)
        if not records:
            return 0.0
            
        similarities = []
        weights = []
        source_weights = {
            'same': 0.90,
            'adjacent': 1.05,
            'global': 1.00
        }
        
        for feature, quality, timestamp, source in records:
            camera_context = {
                'same_camera': source == 'same',
                'adjacent_cameras': source == 'adjacent'
            }
            
            sim = similarity_calculator.calculate_similarity(
                query_feature, feature, quality, camera_context=camera_context
            )
            similarities.append(sim)
            
            time_decay = np.exp(-(time.time() - timestamp) / 3600)
            weight = quality * time_decay * source_weights.get(source, 1.0)
            weights.append(weight)
            
        if not similarities:
            return 0.0
            
        if sum(weights) > 0:
            weighted_sim = np.average(similarities, weights=weights)
        else:
            weighted_sim = np.mean(similarities)
            
        max_sim = max(similarities)
        final_sim = weighted_sim * 0.7 + max_sim * 0.3
        
        return final_sim
    
    def cleanup_old_features(self, max_age: float = 360):
        current_time = time.time()
        to_remove_global = []
        
        for gid, records in self.feature_records.items():
            fresh_records = [r for r in records if current_time - r[2] <= max_age]
            
            if fresh_records:
                self.feature_records[gid] = fresh_records
            else:
                to_remove_global.append(gid)

        for gid in to_remove_global:
            del self.feature_records[gid]

        to_remove_camera = []
        for gid, buckets in self.camera_feature_records.items():
            empty_cameras = []
            for camera_id, records in buckets.items():
                fresh_records = [r for r in records if current_time - r[2] <= max_age]
                if fresh_records:
                    buckets[camera_id] = fresh_records
                else:
                    empty_cameras.append(camera_id)

            for camera_id in empty_cameras:
                del buckets[camera_id]

            if not buckets:
                to_remove_camera.append(gid)

        for gid in to_remove_camera:
            del self.camera_feature_records[gid]

class VehicleReID:
    
    def __init__(self, model_path, config_path, test_path="models/deit_transreid_vehicleID.pth", 
                 matching_threshold=0.6, use_reranking=True, k1=20, k2=6, lambda_value=0.3,
                 camera_topology: Optional[Dict[str, List[str]]] = None,
                 max_features_per_id: int = 6,
                 max_features_per_camera: int = 2):
        try:
            from TransReID.model import make_model
            from TransReID.config import cfg
            self.cfg = cfg
        except ImportError as e:
            logger.error(f"导入TransReID模块失败: {e}")
            raise ImportError("请确保TransReID模块已被正确安装")
        
        if config_path and os.path.exists(config_path):
            try:
                self.cfg.merge_from_file(config_path)
                logger.info(f"从 {config_path} 加载配置成功")
            except Exception as e:
                logger.error(f"从 {config_path} 加载配置失败: {e}")
        
        self.cfg.MODEL.PRETRAIN_PATH = model_path
        self.cfg.TEST.WEIGHT = test_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"增强ReID系统使用设备: {self.device}")
        
        try:
            num_classes = 13164
            camera_num = 0
            view_num = 0
            
            self.model = make_model(self.cfg, num_class=num_classes, 
                                   camera_num=camera_num, view_num=view_num)
            
            if os.path.exists(model_path):
                self.model.load_param(self.cfg.TEST.WEIGHT)
                logger.info(f"从 {self.cfg.TEST.WEIGHT} 加载模型权重成功")
            else:
                logger.warning(f"模型权重 {model_path} 不存在")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"初始化模型错误: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
        
        self.transform = T.Compose([
            T.Resize(self.cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        ])
        
        self.similarity_calculator = EnhancedSimilarityCalculator()
        self.threshold_manager = AdaptiveThresholdManager(matching_threshold)
        self.quality_assessor = FeatureQualityAssessor()
        self.multi_feature_manager = MultiFeatureManager(
            max_features_per_id=max_features_per_id,
            max_features_per_camera=max_features_per_camera
        )
        self.camera_topology = camera_topology or {}
        self.same_camera_similarity_penalty = 0.96
        self.same_camera_threshold_boost = 0.05
        self.same_camera_ambiguity_margin = 0.03
        
        self.gallery = {}
        self.vehicle_metadata = {}
        self.next_id = 1
        
        self.use_reranking = use_reranking
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value
        
        self.performance_stats = {
            'total_extractions': 0,
            'successful_matches': 0,
            'new_assignments': 0,
            'reranking_uses': 0,
            'quality_rejections': 0
        }
        
        if use_reranking:
            logger.info(f"启用增强重排序 (k1={k1}, k2={k2}, lambda={lambda_value})")
        
        logger.info(f"增强车辆ReID系统初始化完成，基础阈值: {matching_threshold}")
    
    def _load_model_weights(self, weight_path):
        state_dict = torch.load(weight_path)
        model_dict = self.model.state_dict()
        
        filtered_state_dict = {}
        for k, v in state_dict.items():
            k_new = k.replace('module.', '')
            
            if k_new in model_dict and v.size() == model_dict[k_new].size():
                filtered_state_dict[k_new] = v
            else:
                logger.debug(f"跳过权重 {k}，尺寸不匹配或在模型中不存在")
        
        self.model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(f"成功加载 {len(filtered_state_dict)}/{len(state_dict)} 层权重参数")

    def extract_features(self, vehicle_patch):
        if vehicle_patch is None or self.model is None:
            return None
            
        self.performance_stats['total_extractions'] += 1
        
        try:
            vehicle_patch_rgb = vehicle_patch[:, :, ::-1]
            img = Image.fromarray(vehicle_patch_rgb)
            
            img_tensor = self.transform(img).unsqueeze(0)
            
            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                cam_label = torch.zeros(1, dtype=torch.long).to(self.device)
                view_label = torch.zeros(1, dtype=torch.long).to(self.device)
                
                if hasattr(self.model, 'forward_jpm'):
                    features = self.model.forward_jpm(img_tensor, cam_label, view_label)
                else:
                    features = self.model(img_tensor, cam_label=cam_label, view_label=view_label)
                
                features_np = features.cpu().numpy()[0]
                
                norm = np.linalg.norm(features_np)
                if norm > 0:
                    features_np = features_np / norm
                
            return features_np
        except Exception as e:
            logger.error(f"特征提取错误: {e}")
            return None
    
    def get_global_id(self, vehicle_patch, camera_id=None, max_gallery_size=1000):
        features = self.extract_features(vehicle_patch)
        if features is None:
            logger.warning("特征提取失败")
            return None
            
        quality_score = self.quality_assessor.assess_image_quality(vehicle_patch)
        
        if quality_score < 0.2:
            self.performance_stats['quality_rejections'] += 1
            logger.debug(f"图像质量过低 ({quality_score:.3f})，拒绝处理")
            return None
        
        adaptive_threshold = self.threshold_manager.current_threshold
        
        if len(self.gallery) > 0:
            best_id, best_score = self._find_best_match(features, quality_score, 
                                                       camera_id, adaptive_threshold)
            
            if best_id is not None:
                self.multi_feature_manager.add_feature(
                    best_id, features, quality_score, camera_id=camera_id
                )
                
                best_feature = self.multi_feature_manager.get_best_feature(best_id)
                if best_feature is not None:
                    self.gallery[best_id] = best_feature
                
                self.vehicle_metadata[best_id]['last_seen'] = time.time()
                if camera_id is not None:
                    if 'cameras' not in self.vehicle_metadata[best_id]:
                        self.vehicle_metadata[best_id]['cameras'] = set()
                    self.vehicle_metadata[best_id]['cameras'].add(camera_id)
                
                self.performance_stats['successful_matches'] += 1
                logger.debug(f"匹配到车辆ID {best_id}，相似度 {best_score:.3f}，阈值 {adaptive_threshold:.3f}")
                return best_id
        
        if len(self.gallery) >= max_gallery_size:
            self._cleanup_oldest_entry()
        
        new_id = self._create_new_vehicle_entry(features, quality_score, camera_id)
        self.performance_stats['new_assignments'] += 1
        logger.debug(f"创建新车辆ID: {new_id}，质量评分: {quality_score:.3f}")
        return new_id
    
    def _find_best_match(self, query_features: np.ndarray, quality_score: float,
                        camera_id: Optional[str], threshold: float) -> Tuple[Optional[int], float]:
        if self.use_reranking and len(self.gallery) >= 5:
            return self._match_with_enhanced_reranking(query_features, quality_score, 
                                                     camera_id, threshold)
        else:
            return self._match_with_enhanced_similarity(query_features, quality_score, 
                                                      camera_id, threshold)
    
    def _match_with_enhanced_similarity(self, query_features: np.ndarray, quality_score: float,
                                      camera_id: Optional[str], threshold: float) -> Tuple[Optional[int], float]:
        best_id = None
        best_score = 0.0
        best_candidate_threshold = threshold
        best_is_same_camera = False
        second_best_score = 0.0
        
        for gid, _ in self.gallery.items():
            # similarity = np.dot(query_features, gallery_feature)
            
            similarity = self.multi_feature_manager.calculate_multi_similarity(
                query_features,
                gid,
                self.similarity_calculator,
                camera_id=camera_id,
                camera_topology=self.camera_topology
            )
            
            quality_factor = 0.95 + 0.05 * quality_score
            adjusted_similarity = similarity * quality_factor

            candidate_threshold = threshold
            is_same_camera_candidate = False
            if camera_id is not None:
                candidate_meta = self.vehicle_metadata.get(gid, {})
                candidate_cameras = candidate_meta.get('cameras', set())
                if camera_id in candidate_cameras:
                    is_same_camera_candidate = True
                    adjusted_similarity *= self.same_camera_similarity_penalty
                    candidate_threshold += self.same_camera_threshold_boost
                    
            if adjusted_similarity > best_score:
                second_best_score = best_score
                best_score = adjusted_similarity
                best_id = gid
                best_candidate_threshold = candidate_threshold
                best_is_same_camera = is_same_camera_candidate
            elif adjusted_similarity > second_best_score:
                second_best_score = adjusted_similarity

        if best_id is None:
            return None, 0.0

        if best_is_same_camera and (best_score - second_best_score) < self.same_camera_ambiguity_margin:
            return None, best_score
        
        if best_score > best_candidate_threshold:
            return best_id, best_score
            
        return None, best_score
    
    def _match_with_enhanced_reranking(self, query_features: np.ndarray, quality_score: float,
                                     camera_id: Optional[str], threshold: float) -> Tuple[Optional[int], float]:
        self.performance_stats['reranking_uses'] += 1
        
        gallery_ids = list(self.gallery.keys())
        gallery_features = []
        
        for gid in gallery_ids:
            best_feature = self.multi_feature_manager.get_best_feature(gid)
            if best_feature is not None:
                gallery_features.append(best_feature)
            else:
                gallery_features.append(self.gallery[gid])
        
        if not gallery_features:
            return None, 0.0
        
        try:
            query_tensor = torch.tensor(query_features).unsqueeze(0)
            gallery_tensor = torch.tensor(np.array(gallery_features))
            
            distmat = re_ranking(query_tensor, gallery_tensor, gallery_tensor,
                               k1=self.k1, k2=self.k2, 
                               lambda_value=self.lambda_value)
            
            distances = distmat[0]
            sorted_indices = np.argsort(distances)
            min_idx = sorted_indices[0]
            min_dist = distances[min_idx]
            
            similarity = np.exp(-min_dist)
            
            best_id = gallery_ids[min_idx]

            second_best_similarity = 0.0
            if len(sorted_indices) > 1:
                second_best_similarity = float(np.exp(-distances[sorted_indices[1]]))

            candidate_threshold = threshold
            is_same_camera_candidate = False
            if camera_id is not None:
                candidate_meta = self.vehicle_metadata.get(best_id, {})
                candidate_cameras = candidate_meta.get('cameras', set())
                if camera_id in candidate_cameras:
                    is_same_camera_candidate = True
                    similarity *= self.same_camera_similarity_penalty
                    candidate_threshold += self.same_camera_threshold_boost

            if is_same_camera_candidate and (similarity - second_best_similarity) < self.same_camera_ambiguity_margin:
                return None, similarity
            
            if similarity > candidate_threshold:
                return best_id, similarity
                
        except Exception as e:
            logger.error(f"重排序匹配错误: {e}")
            return self._match_with_enhanced_similarity(query_features, quality_score, 
                                                      camera_id, threshold)
        
        return None, 0.0
    
    def _create_new_vehicle_entry(self, features: np.ndarray, quality_score: float, 
                                camera_id: Optional[str]) -> int:
        new_id = self.next_id
        self.next_id += 1
        
        self.multi_feature_manager.add_feature(
            new_id, features, quality_score, camera_id=camera_id
        )
        
        self.gallery[new_id] = features
        
        self.vehicle_metadata[new_id] = {
            'first_seen': time.time(),
            'last_seen': time.time(),
            'cameras': {camera_id} if camera_id is not None else set(),
            'initial_quality': quality_score
        }
        
        return new_id
    
    def _cleanup_oldest_entry(self):
        if not self.vehicle_metadata:
            return
            
        oldest_id = min(self.vehicle_metadata, 
                       key=lambda x: self.vehicle_metadata[x]['last_seen'])
        
        if oldest_id in self.gallery:
            del self.gallery[oldest_id]
        if oldest_id in self.vehicle_metadata:
            del self.vehicle_metadata[oldest_id]
        
        logger.debug(f"清理最旧车辆ID: {oldest_id}")

    def update_gallery(self, global_id, vehicle_patch=None, features=None,
                      camera_id: Optional[str] = None):
        if global_id not in self.gallery:
            logger.warning(f"尝试更新不存在的车辆ID: {global_id}")
            return False
        
        if features is None and vehicle_patch is not None:
            features = self.extract_features(vehicle_patch)
        
        if features is None:
            logger.warning(f"无法获取特征用于更新ID {global_id}")
            return False
        
        try:
            quality_score = 0.5
            if vehicle_patch is not None:
                quality_score = self.quality_assessor.assess_image_quality(vehicle_patch)
            
            current_features = self.gallery[global_id]
            
            similarity = self.similarity_calculator.calculate_similarity(
                features, current_features, quality_score
            )
            
            adaptive_threshold = self.threshold_manager.get_threshold()
            verification_threshold = adaptive_threshold * 0.7
            
            if similarity < verification_threshold:
                logger.warning(f"ID {global_id} 特征更新被拒绝，相似度 {similarity:.3f} < 验证阈值 {verification_threshold:.3f}")
                return False
            
            self.multi_feature_manager.add_feature(
                global_id, features, quality_score, camera_id=camera_id
            )
            
            best_feature = self.multi_feature_manager.get_best_feature(global_id)
            if best_feature is not None:
                self.gallery[global_id] = best_feature
            else:
                alpha = 0.8
                updated_features = alpha * current_features + (1 - alpha) * features
                
                norm = np.linalg.norm(updated_features)
                if norm > 0:
                    updated_features = updated_features / norm
                
                self.gallery[global_id] = updated_features
            
            if global_id in self.vehicle_metadata:
                self.vehicle_metadata[global_id]['last_seen'] = time.time()
            
            logger.debug(f"成功更新车辆ID {global_id} 的特征，相似度: {similarity:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"更新车辆ID {global_id} 时出错: {e}")
            return False

    def cleanup_gallery(self, max_age=360):
        current_time = time.time()
        to_remove = []
        
        for gid, metadata in self.vehicle_metadata.items():
            if current_time - metadata['last_seen'] > max_age:
                to_remove.append(gid)
        
        for gid in to_remove:
            if gid in self.gallery:
                del self.gallery[gid]
            if gid in self.vehicle_metadata:
                del self.vehicle_metadata[gid]
        
        self.multi_feature_manager.cleanup_old_features(max_age)
        
        if to_remove:
            logger.info(f"移除了 {len(to_remove)} 个过期的特征库条目（{', GID'.join(map(str, to_remove))}）")

    def get_gallery_stats(self):
        stats = {
            'total_vehicles': len(self.gallery),
            'cameras': set(),
            'cross_camera_vehicles': 0,
        }
        
        for gid, metadata in self.vehicle_metadata.items():
            if 'cameras' in metadata:
                cameras = metadata['cameras']
                stats['cameras'].update(cameras)
                if len(cameras) > 1:
                    stats['cross_camera_vehicles'] += 1
        
        stats['cameras'] = len(stats['cameras'])
        return stats
        
    def debug_folder(self, image_folder, output_folder=None, image_exts=('.jpg', '.jpeg', '.png', '.bmp'), visualize=True):
        if not os.path.exists(image_folder):
            logger.error(f"输入文件夹不存在: {image_folder}")
            return None
        
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
        
        image_files = []
        for ext in image_exts:
            image_files.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(image_folder, f"*{ext.upper()}")))
        
        image_files = sorted(image_files)
        
        if not image_files:
            logger.error(f"在 {image_folder} 中未找到图片文件")
            return None
        
        logger.info(f"找到 {len(image_files)} 个图片文件")
        
        self.gallery = {}
        self.vehicle_metadata = {}
        self.next_id = 1
        
        results = {
            'total_images': len(image_files),
            'unique_vehicles': 0,
            'id_to_files': {},
            'file_to_id': {},
            'image_features': {}
        }
        
        valid_image_paths = []
        for img_path in tqdm(image_files, desc="处理图片"):
            try:
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    logger.warning(f"无法读取图片: {img_path}")
                    continue
                
                valid_image_paths.append(img_path)
                file_name = os.path.basename(img_path)
                
                features = self.extract_features(img_cv)
                if features is None:
                    logger.warning(f"无法提取图片特征: {img_path}")
                    continue
                
                results['image_features'][file_name] = features
                
                global_id = self.get_global_id(img_cv)
                
                if global_id is None:
                    logger.warning(f"无法为图片分配ID: {img_path}")
                    continue
                
                results['file_to_id'][file_name] = global_id
                
                if global_id not in results['id_to_files']:
                    results['id_to_files'][global_id] = []
                results['id_to_files'][global_id].append(file_name)
                
                if output_folder is not None:
                    id_folder = os.path.join(output_folder, f"id_{global_id}")
                    os.makedirs(id_folder, exist_ok=True)
                    shutil.copy(img_path, os.path.join(id_folder, file_name))
                
            except Exception as e:
                logger.error(f"处理图片 {img_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        results['unique_vehicles'] = len(results['id_to_files'])
        
        if visualize and len(self.gallery) > 1 and output_folder is not None:
            self._generate_similarity_matrix(results, output_folder)
        
        if output_folder is not None:
            self._generate_summary_report(results, os.path.join(output_folder, "report.txt"))
        
        logger.info(f"处理完成: 共 {results['total_images']} 张图片, 识别出 {results['unique_vehicles']} 个不同车辆")
        
        if self.use_reranking:
            logger.info(f"使用了特征重排序进行匹配，k1={self.k1}, k2={self.k2}, lambda={self.lambda_value}")
        
        return results
    
    def _generate_similarity_matrix(self, results, output_folder):
        try:
            ids = list(self.gallery.keys())
            features = [self.gallery[id] for id in ids]
            
            n = len(ids)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        sim_matrix[i, j] = 1.0
                    else:
                        sim_matrix[i, j] = np.dot(features[i], features[j])
            
            results['similarities'] = {
                'ids': ids,
                'matrix': sim_matrix
            }
            
            self._visualize_similarity_matrix(sim_matrix, ids, 
                                             os.path.join(output_folder, "similarity_matrix.png"),
                                             title="车辆ID相似度矩阵")
            
            if len(results['image_features']) <= 50:
                self._generate_image_similarity_matrix(results, output_folder)
            
        except Exception as e:
            logger.error(f"生成相似度矩阵时出错: {e}")
    
    def _generate_image_similarity_matrix(self, results, output_folder):
        try:
            image_names = list(results['image_features'].keys())
            features = [results['image_features'][name] for name in image_names]
            
            n = len(image_names)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        sim_matrix[i, j] = 1.0
                    else:
                        sim_matrix[i, j] = np.dot(features[i], features[j])
            
            results['image_similarities'] = {
                'image_names': image_names,
                'matrix': sim_matrix
            }
            
            self._visualize_similarity_matrix(sim_matrix, image_names, 
                                            os.path.join(output_folder, "image_similarity_matrix.png"),
                                            title="图像相似度矩阵")
        except Exception as e:
            logger.error(f"生成图像相似度矩阵时出错: {e}")
            
    def _visualize_similarity_matrix(self, similarity_matrix, ids, output_path, title='Vehicle ID Similarity Matrix'):
        max_display = min(50, len(ids))
        if len(ids) > max_display:
            ids = ids[:max_display]
            similarity_matrix = similarity_matrix[:max_display, :max_display]
        
        plt.figure(figsize=(12, 10))
        plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Similarity')
        
        if len(ids) <= 30:
            plt.xticks(range(len(ids)), [f"{id}" for id in ids], rotation=90, fontsize=8)
            plt.yticks(range(len(ids)), [f"{id}" for id in ids], fontsize=8)
        else:
            plt.xticks([])
            plt.yticks([])
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _generate_summary_report(self, results, output_path):
        with open(output_path, 'w') as f:
            f.write("# Vehicle ReID Summary Report\n\n")
            f.write(f"Total images processed: {results['total_images']}\n")
            f.write(f"Unique vehicles identified: {results['unique_vehicles']}\n\n")
            
            f.write("## ID to Files Mapping\n\n")
            for gid, files in sorted(results['id_to_files'].items()):
                f.write(f"ID {gid}: {len(files)} images\n")
                for file in files:
                    f.write(f"  - {file}\n")
                f.write("\n")
            
            if 'similarities' in results and isinstance(results['similarities'], dict):
                if 'ids' in results['similarities'] and 'matrix' in results['similarities']:
                    f.write("## Cross-ID Similarities\n\n")
                    ids = results['similarities']['ids']
                    matrix = results['similarities']['matrix']
                    
                    f.write("| ID | " + " | ".join([f"ID {id}" for id in ids]) + " |\n")
                    f.write("|" + "---|" * (len(ids) + 1) + "\n")
                    
                    for i, id1 in enumerate(ids):
                        row = [f"ID {id1}"]
                        for j, id2 in enumerate(ids):
                            row.append(f"{matrix[i, j]:.3f}")
                        f.write("| " + " | ".join(row) + " |\n")
                    
                    f.write("\n")
            
            if 'reranking_sim_matrix' in results and isinstance(results['reranking_sim_matrix'], dict):
                f.write("## Re-ranking Image Similarities\n\n")
                
                image_names = results['reranking_sim_matrix'].get('image_names', [])
                matrix = results['reranking_sim_matrix'].get('matrix', None)
                
                if len(image_names) > 0 and matrix is not None:
                    max_display = min(20, len(image_names))
                    if len(image_names) > max_display:
                        f.write(f"注意：图像数量过多({len(image_names)}张)，只显示前 {max_display} 个图像的相似度\n\n")
                        image_names = image_names[:max_display]
                        matrix = matrix[:max_display, :max_display]
                    
                    f.write("| Image | " + " | ".join([f"{name[:10]}..." if len(name) > 12 else name for name in image_names]) + " |\n")
                    f.write("|" + "---|" * (len(image_names) + 1) + "\n")
                    
                    for i, img1 in enumerate(image_names):
                        row = [f"{img1[:10]}..." if len(img1) > 12 else img1]
                        for j, _ in enumerate(image_names):
                            row.append(f"{matrix[i, j]:.3f}")
                        f.write("| " + " | ".join(row) + " |\n")
                    
                    f.write("\n")
                    
                    f.write("## 高相似度图像对 (>0.8)\n\n")
                    high_sim_pairs = []
                    for i in range(len(image_names)):
                        for j in range(i+1, len(image_names)):
                            if matrix[i, j] > 0.8:
                                img1 = image_names[i]
                                img2 = image_names[j]
                                id1 = results['file_to_id'].get(img1, "Unknown")
                                id2 = results['file_to_id'].get(img2, "Unknown")
                                high_sim_pairs.append((img1, img2, matrix[i, j], id1, id2))
                    
                    high_sim_pairs.sort(key=lambda x: x[2], reverse=True)
                    
                    if high_sim_pairs:
                        f.write("| Image 1 | Image 2 | Similarity | ID 1 | ID 2 | Same ID |\n")
                        f.write("|---|---|---|---|---|---|\n")
                        for img1, img2, sim, id1, id2 in high_sim_pairs:
                            same_id = "Yes" if id1 == id2 else "No"
                            f.write(f"| {img1} | {img2} | {sim:.3f} | {id1} | {id2} | {same_id} |\n")
                    else:
                        f.write("没有找到相似度大于0.8的图像对\n")
                else:
                    f.write("无有效的重排序相似度数据\n")
                    
            elif 'image_similarities' in results:
                f.write("## Image-to-Image Similarities\n\n")
                
                image_names = results.get('image_similarities', {}).get('image_names', [])
                matrix = results.get('image_similarities', {}).get('matrix', None)
                
                if len(image_names) > 0 and matrix is not None:
                    max_display = min(30, len(image_names))
                    if len(image_names) > max_display:
                        f.write(f"注意：图像数量过多，只显示前 {max_display} 个图像的相似度\n\n")
                        image_names = image_names[:max_display]
                        matrix = matrix[:max_display, :max_display]
                    
                    f.write("| Image | " + " | ".join([f"{name[:10]}..." if len(name) > 12 else name for name in image_names]) + " |\n")
                    f.write("|" + "---|" * (len(image_names) + 1) + "\n")
                    
                    for i, img1 in enumerate(image_names):
                        row = [f"{img1[:10]}..." if len(img1) > 12 else img1]
                        for j, _ in enumerate(image_names):
                            row.append(f"{matrix[i, j]:.3f}")
                        f.write("| " + " | ".join(row) + " |\n")
                    
                    f.write("\n")
                    
                    f.write("## 高相似度图像对 (>0.8)\n\n")
                    high_sim_pairs = []
                    for i in range(len(image_names)):
                        for j in range(i+1, len(image_names)):
                            if matrix[i, j] > 0.8:
                                img1 = image_names[i]
                                img2 = image_names[j]
                                id1 = results['file_to_id'].get(img1, "Unknown")
                                id2 = results['file_to_id'].get(img2, "Unknown")
                                high_sim_pairs.append((img1, img2, matrix[i, j], id1, id2))
                    
                    high_sim_pairs.sort(key=lambda x: x[2], reverse=True)
                    
                    if high_sim_pairs:
                        f.write("| Image 1 | Image 2 | Similarity | ID 1 | ID 2 | Same ID |\n")
                        f.write("|---|---|---|---|---|---|\n")
                        for img1, img2, sim, id1, id2 in high_sim_pairs:
                            same_id = "Yes" if id1 == id2 else "No"
                            f.write(f"| {img1} | {img2} | {sim:.3f} | {id1} | {id2} | {same_id} |\n")
                    else:
                        f.write("没有找到相似度大于0.8的图像对\n")
                else:
                    f.write("无有效的图像相似度数据\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle ReID Debug Tool")
    parser.add_argument("--input", type=str, default="data/test_images/111", help="Input folder containing vehicle images")
    parser.add_argument("--output", type=str, default="data/test_images/111", help="Output folder for results")
    parser.add_argument("--model", type=str, default="models/deit_base_distilled_patch16_224-df68dfff.pth", help="Path to model weights")
    parser.add_argument("--config", type=str, default="TransReID/configs/VehicleID/deit_transreid_stride.yml", help="Path to config file")
    parser.add_argument("--threshold", type=float, default=0.6, help="Matching threshold (0-1)")
    parser.add_argument("--reranking", action="store_true", help="Enable feature re-ranking")
    parser.add_argument("--k1", type=int, default=20, help="Re-ranking parameter k1")
    parser.add_argument("--k2", type=int, default=6, help="Re-ranking parameter k2")
    parser.add_argument("--lambda_value", type=float, default=0.3, help="Re-ranking parameter lambda")
    
    args = parser.parse_args()
    
    reid_system = VehicleReID(
        model_path=args.model,
        config_path=args.config,
        matching_threshold=args.threshold,
        use_reranking=args.reranking,
        k1=args.k1,
        k2=args.k2,
        lambda_value=args.lambda_value
    )
    
    reid_system.debug_folder(args.input, args.output)