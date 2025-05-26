"""
바운딩 박스 변환 유틸리티

좌표 변환을 위한 함수들을 제공합니다.
"""

import torch


def box_cxcywh_to_xyxy(x):
    """
    (cx, cy, w, h) 형식에서 (x1, y1, x2, y2) 형식으로 변환합니다.
    
    Args:
        x: 박스 좌표 텐서 [..., 4]
    
    Returns:
        변환된 박스 좌표 텐서 [..., 4]
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    (x1, y1, x2, y2) 형식에서 (cx, cy, w, h) 형식으로 변환합니다.
    
    Args:
        x: 박스 좌표 텐서 [..., 4]
    
    Returns:
        변환된 박스 좌표 텐서 [..., 4]
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """
    두 세트의 박스 간 IoU(Intersection over Union) 계산
    
    Args:
        boxes1: 첫 번째 박스 세트 [..., 4] (x1,y1,x2,y2 형식)
        boxes2: 두 번째 박스 세트 [..., 4] (x1,y1,x2,y2 형식)
    
    Returns:
        iou: 각 박스 페어의 IoU 값
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def box_area(boxes):
    """
    박스의 면적 계산
    
    Args:
        boxes: 박스 좌표 텐서 [..., 4] (x1,y1,x2,y2 형식)
    
    Returns:
        각 박스의 면적
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def generalized_box_iou(boxes1, boxes2):
    """
    일반화된 IoU 계산
    
    Args:
        boxes1: 첫 번째 박스 세트 [..., 4] (x1,y1,x2,y2 형식)
        boxes2: 두 번째 박스 세트 [..., 4] (x1,y1,x2,y2 형식)
    
    Returns:
        giou: 일반화된 IoU 값
    """
    # IoU 계산
    iou = box_iou(boxes1, boxes2)
    
    # 포함 상자(Enclosing box) 구하기
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    enclosing_area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # GIoU 계산
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - iou * (area1[:, None] + area2)
    
    giou = iou - (enclosing_area - union) / enclosing_area
    
    return giou 