"""
D-FINE 모델의 손실 함수

디퓨전 기반 객체 감지를 위한 손실 함수를 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class DiffusionCriterion(nn.Module):
    """
    D-FINE 디퓨전 기반 객체 감지 모델을 위한 손실 함수
    
    Args:
        diffusion_steps: 디퓨전 프로세스의 총 스텝 수
        weight_dict: 각 손실 항목의 가중치 딕셔너리
    """
    
    def __init__(self, diffusion_steps=100, weight_dict=None):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        
        # 기본 가중치 설정
        if weight_dict is None:
            self.weight_dict = {
                'loss_ce': 1.0,       # 분류 손실
                'loss_giou': 2.0,      # GIoU 손실
                'loss_bbox': 5.0,      # L1 박스 좌표 손실
                'loss_diffusion': 1.0  # 디퓨전 손실
            }
        else:
            self.weight_dict = weight_dict
            
        # 바이어스 스케줄 설정 (디퓨전 프로세스 제어)
        # 확산 프로세스 내에서 노이즈 스케일을 제어하는 파라미터
        betas = torch.linspace(0.0001, 0.02, diffusion_steps)
        
        # 알파 값 계산 (1 - beta)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 스케일링 파라미터 사전 계산
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # 후보자 매칭을 위한 Hungarian 매처
        self.matcher = DiffusionHungarianMatcher()
        
    def forward(self, outputs, targets):
        """
        손실 함수 계산
        
        Args:
            outputs: 모델 출력 (pred_logits, pred_boxes 포함)
            targets: 타겟 데이터 리스트
            
        Returns:
            총 손실 값 (텐서)
        """
        # 출력과 타겟 구조 확인
        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs
        
        # 디바이스 확인
        device = outputs['pred_logits'].device
        
        # 배치 크기
        batch_size = len(targets)
        
        # 예측 로짓과 박스
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
        pred_boxes = outputs['pred_boxes']    # [B, num_queries, 4]
        
        # 모든 타겟 텐서를 모델과 동일한 디바이스로 이동
        for b in range(batch_size):
            if 'boxes' in targets[b] and len(targets[b]['boxes']) > 0:
                targets[b]['boxes'] = targets[b]['boxes'].to(device)
            if 'labels' in targets[b]:
                targets[b]['labels'] = targets[b]['labels'].to(device)
            if 'time' in targets[b]:
                targets[b]['time'] = targets[b]['time'].to(device)
                
        # 매칭 인덱스 계산 (Hungarian 알고리즘 사용)
        indices = self.matcher(outputs, targets)
        
        # 손실 합계 초기화
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 각 배치 항목에 대해 손실 계산
        for b in range(batch_size):
            if len(targets[b]['boxes']) == 0:
                # 타겟이 없는 경우 처리
                continue
                
            # 매칭 인덱스 가져오기
            idx_src, idx_tgt = indices[b]
            
            if len(idx_src) == 0:
                # 매칭된 항목이 없는 경우 건너뛰기
                continue
                
            # 클래스 손실 계산 - 매칭된 항목만 사용
            tgt_classes = targets[b]['labels'][idx_tgt]
            
            # 선택된 예측만 사용
            selected_pred_logits = pred_logits[b, idx_src]
            
            # 배치 크기 문제 확인
            if selected_pred_logits.size(0) != tgt_classes.size(0):
                # 크기를 맞추기 위해 더 작은 크기로 잘라냄
                min_size = min(selected_pred_logits.size(0), tgt_classes.size(0))
                selected_pred_logits = selected_pred_logits[:min_size]
                tgt_classes = tgt_classes[:min_size]
            
            # 클래스 인덱스를 직접 타겟으로 사용 (one-hot 변환 없이)
            if len(tgt_classes) > 0:
                loss_ce = F.cross_entropy(selected_pred_logits, tgt_classes)
            else:
                loss_ce = torch.tensor(0.0, device=device)
            
            # 박스 좌표 손실 계산
            src_boxes = pred_boxes[b, idx_src]
            tgt_boxes = targets[b]['boxes'][idx_tgt]
            
            # L1 손실
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='sum') / max(len(idx_src), 1)
            
            # GIoU 손실
            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            
            # 디바이스 확인 및 동일하게 맞추기
            if src_boxes_xyxy.device != tgt_boxes_xyxy.device:
                tgt_boxes_xyxy = tgt_boxes_xyxy.to(src_boxes_xyxy.device)
            
            loss_giou = 1 - torch.diag(generalized_box_iou(
                src_boxes_xyxy, tgt_boxes_xyxy
            )).mean()
            
            # 디퓨전 손실 계산 (시간 정보 필요)
            if 'time' in targets[b]:
                t = targets[b]['time'].item()
                noise_scale = self.get_noise_scale(t)
                
                # 현재 예측과 타겟 간의 노이즈 손실
                # 디퓨전 기반의 객체 감지는 예측값을 타겟으로 점진적으로 변환
                noise = torch.randn_like(src_boxes) * noise_scale
                noised_tgt = tgt_boxes + noise
                
                loss_diffusion = F.mse_loss(src_boxes, noised_tgt, reduction='mean')
            else:
                loss_diffusion = torch.tensor(0.0, device=device)
            
            # 가중치 적용 및 총 손실에 더하기
            loss = loss + (
                self.weight_dict['loss_ce'] * loss_ce +
                self.weight_dict['loss_bbox'] * loss_bbox +
                self.weight_dict['loss_giou'] * loss_giou +
                self.weight_dict['loss_diffusion'] * loss_diffusion
            )
        
        # 배치 크기로 정규화
        loss = loss / max(batch_size, 1)
        
        return loss
    
    def get_noise_scale(self, t):
        """시간 스텝 t에 대한 노이즈 스케일 계산"""
        alpha_t = self.alphas_cumprod[t]
        return torch.sqrt(1 - alpha_t)


class DiffusionHungarianMatcher(nn.Module):
    """
    Hungarian 매칭 알고리즘을 사용한 예측-타겟 매처
    
    비용 매트릭스를 계산하고 최적의 매칭을 찾습니다.
    """
    
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    def forward(self, outputs, targets):
        """
        매칭 인덱스 계산
        
        Args:
            outputs: 모델 출력 (pred_logits, pred_boxes 포함)
            targets: 타겟 데이터 리스트
            
        Returns:
            매칭 인덱스 리스트 [(src_idx, tgt_idx), ...]
        """
        batch_size = len(targets)
        indices = []
        
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
        pred_boxes = outputs['pred_boxes']    # [B, num_queries, 4]
        device = pred_boxes.device  # 모델 디바이스 가져오기
        
        # 각 배치 항목에 대해 매칭 수행
        for b in range(batch_size):
            # 타겟 클래스 및 박스
            tgt_labels = targets[b]['labels']
            tgt_boxes = targets[b]['boxes']
            
            if len(tgt_boxes) == 0:
                # 타겟이 없는 경우 빈 매칭 반환
                indices.append(([], []))
                continue
            
            # 타겟 텐서를 모델과 동일한 디바이스로 이동
            tgt_labels = tgt_labels.to(device)
            tgt_boxes = tgt_boxes.to(device)
            
            # 분류 비용 계산
            out_prob = F.softmax(pred_logits[b], dim=-1)
            
            # 크기 불일치 문제 해결: out_prob는 [num_queries, num_classes+1]이고
            # tgt_labels의 값들이 num_classes+1 범위 내에 있는지 확인
            valid_labels = tgt_labels < out_prob.size(1)
            if not torch.all(valid_labels):
                # 유효한 라벨만 선택
                tgt_labels = tgt_labels[valid_labels]
                tgt_boxes = tgt_boxes[valid_labels]
                
                if len(tgt_labels) == 0:
                    # 유효한 라벨이 없으면 빈 매칭 반환
                    indices.append(([], []))
                    continue
            
            out_prob = out_prob[:, tgt_labels]  # [num_queries, num_targets]
            
            # 부의 로그 확률 = 비용
            cost_class = -out_prob
            
            # 박스 좌표 비용 계산
            cost_bbox = torch.cdist(
                pred_boxes[b], tgt_boxes, p=1)  # [num_queries, num_targets]
            
            # GIoU 비용 계산
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
            tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            
            # 디바이스 확인 및 동일하게 맞추기
            if pred_boxes_xyxy.device != tgt_boxes_xyxy.device:
                tgt_boxes_xyxy = tgt_boxes_xyxy.to(pred_boxes_xyxy.device)
            
            cost_giou = -generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)
            
            # 최종 비용 매트릭스
            C = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )
            
            # Hungarian 알고리즘으로 최적 매칭 계산
            # 간소화를 위해 탐욕적 매칭 사용
            # 실제로는 scipy.optimize.linear_sum_assignment 사용 권장
            num_queries, num_targets = C.shape
            
            # 비용 매트릭스 복사
            C_copy = C.clone().detach().cpu().numpy()
            
            # 가장 작은 비용을 가진 매칭 탐색 (탐욕적 방법)
            src_indices = []
            tgt_indices = []
            
            for _ in range(min(num_queries, num_targets)):
                # 최소 비용 찾기
                i, j = divmod(C_copy.argmin(), num_targets)
                
                # 이미 매칭된 항목 제외
                if i in src_indices or j in tgt_indices:
                    # 이미 사용된 인덱스는 무한대로 설정
                    C_copy[i, :] = float('inf')
                    C_copy[:, j] = float('inf')
                    continue
                
                src_indices.append(i)
                tgt_indices.append(j)
                
                # 이미 사용된 인덱스는 무한대로 설정
                C_copy[i, :] = float('inf')
                C_copy[:, j] = float('inf')
            
            indices.append((src_indices, tgt_indices))
        
        return indices 