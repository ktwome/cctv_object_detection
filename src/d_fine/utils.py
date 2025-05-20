"""
D-FINE 모델의 유틸리티 기능

EMA(Exponential Moving Average)와 기타 유틸리티 기능을 제공합니다.
"""

import torch
import copy
import random
import numpy as np


class ModelEma:
    """
    모델 가중치의 지수이동평균(Exponential Moving Average)
    
    학습 안정성을 높이고 일반화 성능을 향상시키기 위해 사용됩니다.
    
    Args:
        model: 원본 모델
        decay: EMA 감쇠율 (기본값: 0.9999)
        device: EMA 모델을 저장할 디바이스
    """
    
    def __init__(self, model, decay=0.9999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        
        # 디바이스 설정
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        
        # EMA 모델을 디바이스로 이동
        self.module.to(device)
        
        # 업데이트 횟수 (EMA 보정용)
        self.updates = 0
    
    def update(self, model):
        """
        모델 가중치 업데이트
        
        Args:
            model: 업데이트할 모델 (원본 모델)
        """
        self.updates += 1
        d = self.decay
        
        # 초기 단계에서는 빠른 수렴을 위해 더 작은 decay 사용
        if self.updates < 2000:
            d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        
        with torch.no_grad():
            # 모델의 모든 가중치 업데이트
            for ema_p, p in zip(self.module.parameters(), model.parameters()):
                ema_p.copy_(d * ema_p + (1 - d) * p)
            
            # 버퍼도 업데이트 (배치 통계 등)
            for ema_b, b in zip(self.module.buffers(), model.buffers()):
                ema_b.copy_(d * ema_b + (1 - d) * b)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, 
                    warmup_epochs=0, start_warmup_value=0):
    """
    코사인 스케줄러
    
    학습률, 가중치 감쇠 등의 하이퍼파라미터를 코사인 스케줄로 조정합니다.
    
    Args:
        base_value: 초기값
        final_value: 최종값
        epochs: 총 에포크 수
        niter_per_ep: 에포크당 반복 횟수
        warmup_epochs: 워밍업 에포크 수
        start_warmup_value: 워밍업 시작값
        
    Returns:
        값 목록
    """
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_iters = warmup_epochs * niter_per_ep
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_schedule.shape[0])
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    
    return np.concatenate([warmup_schedule, schedule])


# 모듈 레벨로 이동된 transforms 함수
def apply_transforms(image, boxes=None, hflip_prob=0.0, resize_range=None, hsv_jitter=0.0):
    """이미지와 박스에 증강 적용 (모듈 레벨 함수)"""
    # 이미지 모양 확인
    if isinstance(image, torch.Tensor):
        # [C, H, W] -> [H, W, C]로 변환
        is_tensor = True
        if image.dim() == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        else:
            raise ValueError("이미지는 [C, H, W] 형태의 텐서여야 합니다.")
    else:
        is_tensor = False
    
    # 박스가 텐서인 경우 NumPy 배열로 변환
    if boxes is not None and isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    # 수평 뒤집기
    if hflip_prob > 0 and random.random() < hflip_prob:
        # 음수 스트라이드 문제를 피하기 위해 복사본 생성
        image = np.ascontiguousarray(image[:, ::-1, :])
        if boxes is not None and len(boxes) > 0:
            # 수평 뒤집기: cx = 1 - cx
            boxes[:, 0] = 1 - boxes[:, 0]
    
    # HSV 색상 지터링
    if hsv_jitter > 0:
        # HSV 지터링 구현 (간략화)
        pass
    
    # 크기 조정
    if resize_range is not None:
        # 크기 조정 구현 (간략화)
        pass
    
    # 텐서로 다시 변환
    if is_tensor:
        # 메모리 연속성 보장
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).permute(2, 0, 1)
        if boxes is not None and len(boxes) > 0:
            boxes = torch.from_numpy(boxes)
    
    return image, boxes


def build_augment(aug_cfg=None):
    """
    데이터 증강 파이프라인 생성
    
    D-FINE 모델을 위한 데이터 증강 파이프라인을 구성합니다.
    여기에서는 간단한 구현만 제공하며, 실제 프로젝트에서는 albumentations 또는
    torchvision.transforms를 사용하는 것이 권장됩니다.
    
    Args:
        aug_cfg: 증강 설정 딕셔너리
    
    Returns:
        증강 파이프라인 함수
    """
    if aug_cfg is None:
        # 증강 없음
        aug_cfg = {}
        
    # 기본적인 데이터 증강 설정
    hflip_prob = aug_cfg.get('horizontal_flip', 0.0)
    resize_range = aug_cfg.get('random_resize', None)
    hsv_jitter = aug_cfg.get('hsv_jitter', 0.0)
    
    # 클래스 기반 변환 객체 반환 (Pickle 가능)
    return TransformWrapper(hflip_prob, resize_range, hsv_jitter)


class TransformWrapper:
    """Pickle 가능한 변환 래퍼 클래스"""
    
    def __init__(self, hflip_prob=0.0, resize_range=None, hsv_jitter=0.0):
        self.hflip_prob = hflip_prob
        self.resize_range = resize_range
        self.hsv_jitter = hsv_jitter
    
    def __call__(self, image, boxes=None):
        return apply_transforms(image, boxes, self.hflip_prob, self.resize_range, self.hsv_jitter) 