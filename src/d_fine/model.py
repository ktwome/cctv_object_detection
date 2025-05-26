"""
D-FINE-B 모델 구현

이 모듈은 D-FINE-B 모델의 핵심 구현을 제공합니다.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class PositionalEncoding(nn.Module):
    """트랜스포머용 위치 인코딩 """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 2D 포지셔널 인코딩용 설정
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: [B, L, D] 형태의 시퀀스
        """
        seq_len = x.size(1)
        return self.pe[:seq_len, :]
    
    def pe1d(self, t):
        """1D 시간 임베딩 계산"""
        t = t.long()
        return self.pe[t]


class FeatureExtractor(nn.Module):
    """ResNet 백본에서 특징 추출 클래스"""
    
    def __init__(self, backbone, hidden_dim=256):
        super().__init__()
        self.backbone = backbone
        
        # ResNet 레이어 이름과 출력 채널 수
        self.layer_info = {
            'layer1': 64,   # ResNet-18의 layer1 출력 채널
            'layer2': 128,  # ResNet-18의 layer2 출력 채널
            'layer3': 256,  # ResNet-18의 layer3 출력 채널
            'layer4': 512   # ResNet-18의 layer4 출력 채널
        }
        
        # 각 레이어의 특징을 동일한 차원으로 투영하는 컨볼루션
        self.projections = nn.ModuleDict({
            name: nn.Conv2d(channels, hidden_dim, kernel_size=1)
            for name, channels in self.layer_info.items()
        })
        
        # 백본의 초기 레이어
        self.initial_layers = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
    
    def forward(self, x):
        # 초기 레이어 통과
        x = self.initial_layers(x)
        
        # 각 레이어에서 특징 추출 및 투영
        features = {}
        for i, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            layer = getattr(self.backbone, layer_name)
            x = layer(x)
            features[layer_name] = self.projections[layer_name](x)
        
        return features


class HybridEncoder(nn.Module):
    """D-FINE을 위한 Hybrid Encoder 구현 (개선된 버전)"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 다양한 스케일의 특징을 융합하는 변환기
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        """
        features: 백본에서 추출한 다양한 레이어의 특징 (딕셔너리)
        """
        # 모든 특징을 가장 작은 해상도(layer4)에 맞게 리사이즈
        target_size = features['layer4'].shape[-2:]
        
        aligned_features = []
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            feat = features[name]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 특징 결합
        fused = torch.cat(aligned_features, dim=1)
        return self.fusion(fused)


class DFineDecoder(nn.Module):
    """D-FINE의 디퓨전 기반 디코더"""
    
    def __init__(self, hidden_dim=256, num_classes=80, num_queries=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # 쿼리 임베딩 초기화 (학습 가능)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 디코더 트랜스포머 층
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        
        # 예측 헤드
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # (cx, cy, w, h)
        )
        
        # 디퓨전 컨디셔닝 임베딩
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 포지셔널 인코딩
        self.pos_encoder = PositionalEncoding(hidden_dim)
    
    def forward(self, features, targets=None):
        """
        features: 인코더에서 나온 특징 맵 [B, C, H, W]
        targets: 학습 시 타겟 정보 (딕셔너리)
        """
        B = features.shape[0]
        H, W = features.shape[-2:]
        
        # 인코더 특징을 시퀀스로 변환 [B, H*W, C]
        memory = features.flatten(2).permute(0, 2, 1)
        
        # 메모리에 포지셔널 인코딩 추가
        pos_embed = self.pos_encoder(memory)
        memory = memory + pos_embed
        
        # 쿼리 임베딩 준비
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # 시간 정보 임베딩 (학습 시)
        if targets is not None and len(targets) > 0:
            # 모든 타겟에 시간 키가 있는지 확인
            has_time = all('time' in target for target in targets)
            if has_time:
                try:
                    # 각 배치 항목에서 시간 추출
                    times = torch.cat([target['time'] for target in targets])
                    time_embed = self.time_embed(self.pos_encoder.pe1d(times))
                    
                    # 배치 내 모든 쿼리에 시간 임베딩 추가
                    expanded_time = time_embed.unsqueeze(1).expand(-1, self.num_queries, -1)
                    query_embed = query_embed + expanded_time
                except (RuntimeError, KeyError, IndexError) as e:
                    print(f"[경고] 시간 임베딩 처리 중 오류 발생: {e}")
                    # 오류 발생 시 시간 임베딩 건너뛰기
                    pass
        
        # 타겟 시퀀스 초기화 - 쿼리 임베딩을 직접 사용
        tgt = query_embed
        
        # 표준 PyTorch TransformerDecoder API 사용
        # query_pos 매개변수 대신 tgt에 포지셔널 정보를 직접 포함
        decoder_out = self.decoder(
            tgt,                         # 타겟 시퀀스 (쿼리)
            memory,                      # 메모리 (인코더 출력)
            tgt_mask=None,               # 타겟 마스크 (필요 없음)
            memory_mask=None,            # 메모리 마스크 (필요 없음)
            tgt_key_padding_mask=None,   # 타겟 패딩 마스크 (필요 없음)
            memory_key_padding_mask=None # 메모리 패딩 마스크 (필요 없음)
        )
        
        # 박스 좌표와 클래스 로짓 예측
        pred_logits = self.class_embed(decoder_out)  # [B, num_queries, num_classes+1]
        pred_boxes = self.bbox_embed(decoder_out).sigmoid()  # [B, num_queries, 4]
        
        # 훈련 모드
        if self.training and targets is not None:
            return {
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes,
                # 추가 정보는 loss 계산에 사용
            }
        
        # 추론 모드
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}


class DFineB(nn.Module):
    """
    D-FINE-B 모델 구현
    
    D-FINE의 D-FINE-B 변형을 구현한 클래스입니다.
    """
    
    def __init__(self, backbone_pretrained=True, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        # ResNet-18 백본 (메모리 요구 사항을 고려해 ResNet-18 사용)
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if backbone_pretrained:
                weights = ResNet18_Weights.DEFAULT
                self.backbone = resnet18(weights=weights)
            else:
                self.backbone = resnet18(weights=None)
        except ImportError:
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=backbone_pretrained)
        
        # 히든 차원 설정
        hidden_dim = 256
        
        # 특징 추출기, 인코더, 디코더 초기화
        self.feature_extractor = FeatureExtractor(self.backbone, hidden_dim)
        self.encoder = HybridEncoder(hidden_dim)
        self.decoder = DFineDecoder(hidden_dim, num_classes, num_queries=100)
    
    def forward(self, x, targets=None):
        """
        전방 전파
        
        Args:
            x: 입력 이미지 텐서 [B, 3, H, W]
            targets: 학습 시 타겟 정보 (딕셔너리 리스트)
            
        Returns:
            학습 시: 손실 계산을 위한 예측 값 딕셔너리
            추론 시: 각 이미지의 검출 결과 (박스, 클래스, 점수)
        """
        # 특징 추출
        features = self.feature_extractor(x)
        
        # 인코더로 특징 융합
        encoded_features = self.encoder(features)
        
        # 디코더로 객체 검출
        outputs = self.decoder(encoded_features, targets)
        
        # 학습 모드
        if self.training and targets is not None:
            return outputs
        
        # 추론 모드
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # 후처리: 클래스별 확률, 박스 변환
        prob = F.softmax(pred_logits, -1)
        scores, labels = prob[..., :-1].max(-1)  # 배경 클래스 제외
        boxes = box_cxcywh_to_xyxy(pred_boxes)
        
        results = {
            'boxes': boxes,      # [B, num_queries, 4]
            'scores': scores,    # [B, num_queries]
            'labels': labels     # [B, num_queries]
        }
        
        return results 