"""
D-FINE 데이터 처리 모듈

COCO 형식의 데이터셋을 처리하고 디퓨전 프로세스를 위한 
데이터 변환 클래스를 제공합니다.
"""

import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_convert
import torch.nn.functional as F

from .box_ops import box_xyxy_to_cxcywh


def get_image_paths(img_root, ann_file):
    """COCO 주석 파일에서 이미지 경로 추출"""
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    img_ids = [img['id'] for img in coco['images']]
    img_paths = {img['id']: os.path.join(img_root, img['file_name']) for img in coco['images']}
    
    # 주석에 해당하는 이미지 ID 확인
    valid_img_ids = set()
    for ann in coco['annotations']:
        valid_img_ids.add(ann['image_id'])
    
    # 주석이 있는 이미지만 반환
    return [img_paths[img_id] for img_id in img_ids if img_id in valid_img_ids]


class CocoDiffusionDataset(Dataset):
    """
    COCO 형식의 데이터셋을 디퓨전 기반 학습을 위해 로드하는 클래스
    
    Args:
        json_file: COCO 형식의 주석 파일 경로
        img_root: 이미지 파일이 있는 루트 디렉터리
        num_classes: 클래스 수
        diffusion_steps: 디퓨전 타임스텝 수
        transforms: 적용할 데이터 증강
        cache_images: 이미지 캐싱 활성화 여부
        max_cache_size: 최대 캐시 크기 (MB 단위, 기본 1GB)
        memory_efficient: 메모리 효율적 모드 활성화 여부
    """
    
    def __init__(self, json_file, img_root, num_classes=80, 
                 diffusion_steps=100, transforms=None,
                 cache_images=True, max_cache_size=1000,
                 memory_efficient=True):
        self.img_root = img_root
        self.json_file = json_file
        self.transforms = transforms
        self.num_classes = num_classes
        self.diffusion_steps = diffusion_steps
        self.cache_images = cache_images
        self.max_cache_size = max_cache_size  # MB 단위
        self.memory_efficient = memory_efficient
        
        # 이미지 캐시 초기화
        self.img_cache = {}  # {img_id: img_tensor}
        self.cache_order = []  # LRU 캐시를 위한 이미지 ID 리스트
        self.current_cache_size = 0  # 현재 캐시 크기 (MB)
        
        # COCO 주석 파일 로드
        with open(json_file, 'r', encoding='utf-8') as f:
            self.coco = json.load(f)
        
        # 이미지 경로와 ID 매핑 생성
        self.img_paths = {img['id']: os.path.join(img_root, img['file_name']) 
                          for img in self.coco['images']}
        
        # 카테고리 ID -> 인덱스 매핑 생성
        self.cat_ids = {cat['id']: i for i, cat in enumerate(self.coco['categories'])}
        
        # 이미지별 주석 생성
        self.img_to_anns = self._build_img_to_anns()
        
        # 유효한 이미지 ID 목록 생성 (주석이 있는 이미지만)
        self.img_ids = [img_id for img_id in self.img_paths.keys() 
                       if img_id in self.img_to_anns and self.img_to_anns[img_id]]
        
        print(f"데이터셋 로드 완료: {json_file} - 이미지 {len(self.img_ids)}개")
        if self.cache_images:
            print(f"이미지 캐싱 활성화: 최대 {self.max_cache_size}MB{' (메모리 효율 모드)' if memory_efficient else ''}")
    
    def _build_img_to_anns(self):
        """이미지 ID별 주석 목록 생성"""
        img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            
            # 필요한 조건 확인
            if (ann.get('ignore', 0) == 1 or 
                ann.get('iscrowd', 0) == 1 or 
                'bbox' not in ann):
                continue
            
            # 카테고리 ID가 매핑에 없는 경우 처리
            if ann['category_id'] not in self.cat_ids:
                continue
                
            # 유효한 주석 추가
            img_to_anns[img_id].append(ann)
            
        return img_to_anns
    
    def _update_cache(self, img_id, img_tensor):
        """LRU 캐시 업데이트"""
        if not self.cache_images:
            return
            
        # 이미지 크기 계산 (MB)
        if self.memory_efficient:
            # 메모리 효율 모드에서는 float32 대신 uint8로 저장 (4배 메모리 절약)
            if img_tensor.dtype == torch.float32:
                img_uint8 = (img_tensor * 255).to(torch.uint8)
                img_size_mb = img_uint8.element_size() * img_uint8.numel() / (1024 * 1024)
                # 캐시에 uint8로 저장
                img_tensor = img_uint8
            else:
                img_size_mb = img_tensor.element_size() * img_tensor.numel() / (1024 * 1024)
        else:
            # 기존 방식
            img_size_mb = img_tensor.element_size() * img_tensor.numel() / (1024 * 1024)
        
        # 캐시에 이미지 추가
        if img_id in self.img_cache:
            # 이미지가 이미 캐시에 있으면 LRU 순서 업데이트
            self.cache_order.remove(img_id)
            self.cache_order.append(img_id)
        else:
            # 새 이미지 추가 전에 캐시 공간 확보
            while (self.current_cache_size + img_size_mb > self.max_cache_size and 
                  self.cache_order):
                # 가장 오래된 이미지 제거
                oldest_id = self.cache_order.pop(0)
                oldest_tensor = self.img_cache.pop(oldest_id)
                oldest_size = oldest_tensor.element_size() * oldest_tensor.numel() / (1024 * 1024)
                self.current_cache_size -= oldest_size
            
            # 새 이미지 추가
            self.img_cache[img_id] = img_tensor
            self.cache_order.append(img_id)
            self.current_cache_size += img_size_mb
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        데이터셋에서 항목 가져오기
        
        Returns:
            dict: 이미지 텐서, 타겟 정보 포함 (박스, 라벨, 시간)
        """
        img_id = self.img_ids[idx]
        
        # 캐시에서 이미지 찾기
        if self.cache_images and img_id in self.img_cache:
            cached_img = self.img_cache[img_id]
            # LRU 순서 업데이트
            self.cache_order.remove(img_id)
            self.cache_order.append(img_id)
            
            # uint8에서 float32로 변환 (필요한 경우)
            if self.memory_efficient and cached_img.dtype == torch.uint8:
                img_tensor = cached_img.float() / 255.0
            else:
                img_tensor = cached_img
        else:
            # 캐시에 없으면 디스크에서 로드
            img_path = self.img_paths[img_id]
            
            # 이미지 로드 (한글 경로 지원)
            try:
                # 메모리 효율적 로딩
                img = Image.open(img_path).convert("RGB")
                
                # 빠른 NumPy 변환
                img_np = np.asarray(img, dtype=np.uint8)
                
                # 이미지를 텐서로 변환 [C, H, W]
                if self.memory_efficient:
                    # uint8 텐서로 변환 후 캐싱
                    img_tensor_uint8 = torch.from_numpy(img_np).permute(2, 0, 1)
                    self._update_cache(img_id, img_tensor_uint8.clone())
                    # float32로 변환하여 반환
                    img_tensor = img_tensor_uint8.float() / 255.0
                else:
                    # 기존 방식
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                    # 캐시에 추가
                    if self.cache_images:
                        self._update_cache(img_id, img_tensor.clone())
                
            except Exception as e:
                # 실패 시 바이너리로 로드 시도
                with open(img_path, 'rb') as f:
                    img_bytes = bytearray(f.read())
                    img_np = np.asarray(img_bytes, dtype=np.uint8)
                    img = Image.fromarray(img_np)
                img_np = np.array(img)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        
        width, height = img_tensor.shape[2], img_tensor.shape[1]
        anns = self.img_to_anns[img_id]
        
        # 주석에서 박스와 클래스 추출
        boxes = []
        labels = []
        
        for ann in anns:
            bbox = ann['bbox']  # COCO 형식: [x, y, width, height]
            
            # 유효한 박스인지 확인
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            
            # COCO 형식(xywh)에서 xyxy 형식으로 변환
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            
            # 정규화된 좌표로 변환
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_ids[ann['category_id']])
        
        # 박스와 라벨을 텐서로 변환
        if not boxes:  # 박스가 없는 경우
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
            # xyxy에서 cxcywh로 변환 (D-FINE 형식)
            boxes = box_xyxy_to_cxcywh(boxes)
        
        # 데이터 증강 적용 (있는 경우)
        if self.transforms is not None:
            img_tensor, boxes = self.transforms(img_tensor, boxes)
        
        # 랜덤 디퓨전 타임스텝 선택
        t = torch.randint(0, self.diffusion_steps, (1,))
        
        target = {
            'boxes': boxes,          # [N, 4] - (cx, cy, w, h) 형식
            'labels': labels,        # [N] - 클래스 인덱스
            'time': t,               # [1] - 디퓨전 타임스텝
            'image_id': img_id       # 이미지 ID (평가용)
        }
        
        return {'image': img_tensor, 'target': target}
    
    def get_class_counts(self):
        """각 클래스별 이미지 수를 계산합니다.
        
        Returns:
            counts: 각 클래스별 이미지 수를 담은 리스트
        """
        # 클래스별 카운트 초기화
        counts = [0] * self.num_classes
        
        # 모든 이미지 ID에 대해 반복
        for img_id in self.img_ids:
            # 이미지에 해당하는 주석 가져오기
            anns = self.img_to_anns.get(img_id, [])
            
            # 해당 이미지의 모든 클래스 추적 (중복 없이)
            img_classes = set()
            
            for ann in anns:
                if 'category_id' in ann and ann['category_id'] in self.cat_ids:
                    cls_idx = self.cat_ids[ann['category_id']]
                    img_classes.add(cls_idx)
            
            # 이미지에 나타난 각 클래스 카운트 증가
            for cls_idx in img_classes:
                counts[cls_idx] += 1
        
        return counts
    
    def get_all_targets(self):
        """모든 타겟(라벨)을 리스트로 반환합니다.
        균등 샘플링을 위해 사용됩니다.
        
        Returns:
            targets: 모든 이미지의 클래스 라벨 리스트
        """
        targets = []
        
        for img_id in self.img_ids:
            anns = self.img_to_anns.get(img_id, [])
            # 이미지에 있는 모든 클래스 중 첫 번째 클래스만 사용
            # (이미지 수준 샘플링을 위해)
            if anns:
                if 'category_id' in anns[0] and anns[0]['category_id'] in self.cat_ids:
                    cls_idx = self.cat_ids[anns[0]['category_id']]
                    targets.append(cls_idx)
                else:
                    # 카테고리가 없는 경우 클래스 0으로 가정
                    targets.append(0)
            else:
                # 주석이 없는 경우 클래스 0으로 가정
                targets.append(0)
        
        return targets


def collate_fn(batch):
    """
    데이터로더를 위한 배치 병합 함수
    
    다양한 객체 수와 크기를 가진 이미지를 배치로 결합합니다.
    서로 다른 크기의 이미지를 처리하기 위해 동일한 크기로 리사이즈합니다.
    """
    # 빈 배치 처리
    if len(batch) == 0:
        return {'image': [], 'target': []}
    
    # 이미지 크기 가져오기
    max_height = max([item['image'].shape[1] for item in batch])
    max_width = max([item['image'].shape[2] for item in batch])
    
    # 이미지 크기가 다른 경우 리사이즈 처리
    resized_images = []
    targets = []
    
    for item in batch:
        img = item['image']
        target = item['target']
        
        # 타겟 유효성 확인
        if 'boxes' not in target or len(target['boxes']) == 0:
            # 최소한 빈 박스와 라벨 배열 제공
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32, device=img.device)
            target['labels'] = torch.zeros(0, dtype=torch.int64, device=img.device)
        
        # 시간 정보 확인
        if 'time' not in target:
            # 시간 정보가 없으면 기본값 추가 (0)
            target['time'] = torch.zeros(1, dtype=torch.int64, device=img.device)
        
        # 원본 크기
        _, h, w = img.shape
        
        if h == max_height and w == max_width:
            # 이미 최대 크기라면 그대로 사용
            resized_images.append(img)
        else:
            # 리사이즈 필요
            # F.interpolate는 배치 차원이 필요하므로 차원 추가
            img_batch = img.unsqueeze(0)
            resized_img = F.interpolate(
                img_batch, 
                size=(max_height, max_width), 
                mode='bilinear', 
                align_corners=False
            )
            # 배치 차원 제거
            resized_images.append(resized_img.squeeze(0))
            
            # 박스 좌표는 정규화되어 있으므로 조정 필요 없음
        
        targets.append(target)
    
    # 리사이즈된 이미지 스택
    images = torch.stack(resized_images)
    
    return {'image': images, 'target': targets}


# 모듈 레벨로 이동된 simple_transforms 함수
def apply_simple_transforms(img, boxes, aug_cfg):
    """이미지와 박스에 증강 적용 (모듈 레벨 함수)"""
    # 원래는 여기서 albumentations와 같은 라이브러리로 증강 적용
    # 지금은 간단히 구현
    
    # 고정 크기 리사이즈 (설정된 경우)
    if 'fixed_size' in aug_cfg:
        target_h, target_w = aug_cfg['fixed_size']
        _, orig_h, orig_w = img.shape
        
        if orig_h != target_h or orig_w != target_w:
            # 이미지 리사이즈
            img = F.interpolate(
                img.unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
    
    # 수평 뒤집기
    if 'horizontal_flip' in aug_cfg and random.random() < aug_cfg['horizontal_flip']:
        img = torch.flip(img, dims=[2])  # 가로 축으로 뒤집기
        # 박스 좌표도 수평 뒤집기
        if len(boxes):
            boxes[:, 0] = 1 - boxes[:, 0]  # cx = 1 - cx
    
    # 크기 조정 (실제 구현에서는 더 복잡)
    if 'random_resize' in aug_cfg:
        # 실제 구현에서는 여기서 크기 조정 로직 구현
        pass
    
    return img, boxes


def build_augment(aug_cfg=None):
    """
    데이터 증강 파이프라인 생성
    
    Args:
        aug_cfg: 증강 설정 딕셔너리
    
    Returns:
        증강 파이프라인 함수
    """
    if aug_cfg is None:
        # 증강 없음 - 기본 함수 반환
        aug_cfg = {}
    
    # 함수 대신 클래스를 사용하여 Pickle 가능하게 만듦
    return TransformPipeline(aug_cfg)


class TransformPipeline:
    """Pickle 가능한 변환 파이프라인 클래스"""
    
    def __init__(self, aug_cfg):
        self.aug_cfg = aug_cfg
    
    def __call__(self, img, boxes=None):
        return apply_simple_transforms(img, boxes, self.aug_cfg) 