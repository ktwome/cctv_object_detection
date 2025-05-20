#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터셋 다운샘플링 스크립트

극심한 클래스 불균형(최대 13만개 : 최소 300개)을 해결하기 위한 다운샘플링 수행
- 클래스별 분포 분석
- 클래스별 데이터 품질 평가
- 전략적 다운샘플링 (상위 6000개 샘플)
- 새로운 균형 잡힌 데이터셋 생성
"""

import os
import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import tempfile
import urllib.request

# 한글 폰트 설정
def set_korean_font():
    """matplotlib에서 한글 폰트를 사용할 수 있도록 설정합니다."""
    system_platform = platform.system()
    
    if system_platform == 'Windows':
        # Windows 환경
        plt.rc('font', family='Malgun Gothic')  # 맑은 고딕
    elif system_platform == 'Darwin':
        # macOS 환경
        plt.rc('font', family='AppleGothic')  # Apple Gothic
    else:
        # Linux 등 기타 환경
        try:
            # 나눔 폰트가 설치되어 있는지 확인
            from matplotlib import font_manager
            font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            nanum_font = None
            
            for font in font_list:
                if 'NanumGothic' in font:
                    nanum_font = font
                    break
            
            if nanum_font:
                plt.rc('font', family='NanumGothic')
            else:
                # 나눔 폰트가 없으면 다운로드
                nanum_font_path = download_nanum_font()
                if nanum_font_path:
                    font_manager.fontManager.addfont(nanum_font_path)
                    plt.rc('font', family='NanumGothic')
                else:
                    print("경고: 한글 폰트를 설치할 수 없습니다. 영문으로 표시됩니다.")
        except Exception as e:
            print(f"경고: 한글 폰트 설정에 실패했습니다. 오류: {e}")
    
    # 음수 표시 문제 해결
    mpl.rcParams['axes.unicode_minus'] = False

def download_nanum_font():
    """나눔고딕 폰트를 다운로드하고 설치합니다."""
    try:
        print("나눔고딕 폰트 다운로드 중...")
        # 나눔고딕 폰트 URL
        nanum_url = "https://github.com/naver/nanumfont/blob/master/NanumGothic.ttf?raw=true"
        
        # 임시 디렉토리에 폰트 다운로드
        temp_dir = tempfile.gettempdir()
        font_path = os.path.join(temp_dir, "NanumGothic.ttf")
        
        # 폰트 다운로드
        urllib.request.urlretrieve(nanum_url, font_path)
        
        print(f"나눔고딕 폰트 다운로드 완료: {font_path}")
        return font_path
    except Exception as e:
        print(f"나눔고딕 폰트 다운로드 실패: {e}")
        return None

# 프로그램 시작 시 한글 폰트 설정
set_korean_font()


def analyze_coco_dataset(json_path):
    """COCO 형식 JSON 파일을 분석하여 클래스별 분포와 이미지당 객체 수 등의 통계를 계산합니다."""
    print(f"COCO 데이터셋 분석 중: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 데이터셋 기본 정보
    num_images = len(coco_data['images'])
    num_annotations = len(coco_data['annotations'])
    num_categories = len(coco_data['categories'])
    
    print(f"데이터셋 정보:")
    print(f" - 이미지 수: {num_images}")
    print(f" - 주석(객체) 수: {num_annotations}")
    print(f" - 카테고리(클래스) 수: {num_categories}")
    
    # 카테고리 ID와 이름 매핑
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 클래스별 객체 수 계산
    class_counts = Counter()
    for ann in coco_data['annotations']:
        class_counts[ann['category_id']] += 1
    
    # 이미지 ID별 주석 그룹화
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # 이미지별 객체 정보 계산
    images_info = {}
    for img in coco_data['images']:
        img_id = img['id']
        anns = image_annotations[img_id]
        
        # 이미지별 객체 수와 클래스 분포
        obj_count = len(anns)
        class_dist = Counter([ann['category_id'] for ann in anns])
        
        # 객체 크기 정보
        areas = [ann['area'] for ann in anns]
        avg_area = np.mean(areas) if areas else 0
        
        images_info[img_id] = {
            'file_name': img['file_name'],
            'obj_count': obj_count,
            'class_dist': class_dist,
            'avg_obj_area': avg_area,
            'annotations': anns
        }
    
    # 클래스별 세부 통계
    class_stats = {}
    for cat_id, cat_name in categories.items():
        # 해당 클래스 객체가 포함된 이미지 수
        images_with_class = sum(1 for img_info in images_info.values() 
                               if cat_id in img_info['class_dist'])
        
        # 해당 클래스 객체의 평균 면적
        areas = [ann['area'] for img_id, img_info in images_info.items() 
                for ann in img_info['annotations'] 
                if ann['category_id'] == cat_id]
        avg_area = np.mean(areas) if areas else 0
        
        class_stats[cat_id] = {
            'name': cat_name,
            'count': class_counts[cat_id],
            'images_count': images_with_class,
            'avg_area': avg_area
        }
    
    return {
        'dataset_info': {
            'num_images': num_images,
            'num_annotations': num_annotations,
            'num_categories': num_categories,
        },
        'categories': categories,
        'class_counts': class_counts,
        'class_stats': class_stats,
        'images_info': images_info,
        'coco_data': coco_data
    }


def visualize_class_distribution(class_stats, output_dir=None):
    """클래스별 분포를 시각화합니다."""
    class_ids = sorted(class_stats.keys())
    class_names = [class_stats[cid]['name'] for cid in class_ids]
    counts = [class_stats[cid]['count'] for cid in class_ids]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(class_names, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('클래스별 객체 수 분포')
    plt.xlabel('클래스')
    plt.ylabel('객체 수')
    plt.tight_layout()
    
    # 막대 위에 값 표시
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha='center', va='bottom', rotation=0)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
        print(f"클래스 분포 시각화 저장됨: {output_dir}/class_distribution.png")
    
    plt.show()


def stratified_downsampling(dataset_analysis, target_samples=6000, max_per_class=None, 
                           min_per_class=None, quality_weight=0.5, output_dir='subset'):
    """
    클래스별로 균형있게 다운샘플링을 수행합니다.
    
    Args:
        dataset_analysis: analyze_coco_dataset 함수의 결과
        target_samples: 최종 이미지 샘플 수
        max_per_class: 클래스당 최대 샘플 수 (None=제한없음)
        min_per_class: 클래스당 최소 샘플 수 (None=제한없음)
        quality_weight: 품질 지표 가중치 (0-1)
        output_dir: 결과 저장 디렉토리
    
    Returns:
        다운샘플링된 COCO 데이터 (이미지 ID 리스트 및 COCO 형식 데이터)
    """
    categories = dataset_analysis['categories']
    class_stats = dataset_analysis['class_stats']
    images_info = dataset_analysis['images_info']
    coco_data = dataset_analysis['coco_data']
    
    num_classes = len(categories)
    print(f"전략적 다운샘플링 수행 중... (대상: {target_samples} 샘플)")
    
    # 클래스별 이미지 ID 그룹화
    class_image_ids = defaultdict(list)
    for img_id, img_info in images_info.items():
        for class_id in img_info['class_dist'].keys():
            class_image_ids[class_id].append(img_id)
    
    # 클래스별 기본 할당량 계산 (클래스당 동일 비율)
    base_quota = target_samples // num_classes
    
    # 클래스별로 중요도 점수 계산 및 샘플 선택
    selected_image_ids = set()
    
    for class_id, stats in class_stats.items():
        available_images = class_image_ids[class_id]
        cat_name = categories[class_id]
        
        # 클래스의 전체 객체 수
        total_count = stats['count']
        
        # 할당량 결정 (기본 할당량 또는 최대/최소 조정)
        quota = base_quota
        if max_per_class:
            quota = min(quota, max_per_class)
        if min_per_class:
            quota = max(quota, min_per_class)
        
        # 실제 가용 이미지보다 많은 할당량 요청시 조정
        actual_quota = min(quota, len(available_images))
        
        print(f"클래스 {cat_name}: 기본 할당량 {base_quota}, 조정 할당량 {actual_quota}, 가용 이미지 {len(available_images)}개")
        
        # 이미지 품질 점수 계산
        image_scores = {}
        for img_id in available_images:
            img_info = images_info[img_id]
            
            # 이 이미지에서의 해당 클래스 객체 수
            class_obj_count = img_info['class_dist'][class_id]
            
            # 객체 면적 평균 (큰 객체가 더 명확히 보임)
            class_areas = [ann['area'] for ann in img_info['annotations'] 
                          if ann['category_id'] == class_id]
            avg_area = np.mean(class_areas) if class_areas else 0
            norm_area = min(avg_area / (img_info['avg_obj_area'] or 1), 3)  # 평균 대비 최대 3배까지 정규화
            
            # 다양성 점수 (이미지에 여러 클래스가 있는 경우 가중치)
            diversity = len(img_info['class_dist']) / num_classes
            
            # 혼잡도 패널티 (너무 많은 객체가 있는 경우 불이익)
            clutter_penalty = 1.0 / (1 + max(0, img_info['obj_count'] - 10) * 0.1)
            
            # 종합 점수 계산 (여러 요소를 조합)
            quality_score = (norm_area * 0.4 + diversity * 0.3 + clutter_penalty * 0.3) * quality_weight
            count_score = (class_obj_count / max(1, img_info['obj_count'])) * (1 - quality_weight)
            
            total_score = quality_score + count_score
            image_scores[img_id] = total_score
        
        # 점수 기준 상위 이미지 선택
        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        top_images = [img_id for img_id, _ in sorted_images[:actual_quota]]
        
        # 선택된 이미지 ID 추가
        selected_image_ids.update(top_images)
        
        print(f"클래스 {cat_name}: {len(top_images)}개 이미지 선택됨")
    
    print(f"총 {len(selected_image_ids)}개 이미지 선택됨")
    
    # 선택된 이미지와 주석만 포함하는 새 COCO 데이터 생성
    downsampled_coco = {
        'images': [img for img in coco_data['images'] if img['id'] in selected_image_ids],
        'annotations': [ann for ann in coco_data['annotations'] 
                       if ann['image_id'] in selected_image_ids],
        'categories': coco_data['categories']
    }
    
    # 선택된 이미지 ID와 새 COCO 데이터 반환
    return {
        'selected_image_ids': selected_image_ids,
        'downsampled_coco': downsampled_coco
    }


def copy_selected_images(selected_image_ids, src_img_dir, dest_img_dir, coco_data):
    """선택된 이미지 파일을 새 디렉토리로 복사합니다."""
    os.makedirs(dest_img_dir, exist_ok=True)
    
    # 이미지 ID와 파일명 매핑
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    print(f"선택된 이미지 복사 중: {src_img_dir} -> {dest_img_dir}")
    for img_id in tqdm(selected_image_ids):
        filename = id_to_filename.get(img_id)
        if not filename:
            continue
            
        src_path = os.path.join(src_img_dir, filename)
        dest_path = os.path.join(dest_img_dir, filename)
        
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
        else:
            print(f"경고: 파일을 찾을 수 없음 - {src_path}")


def create_yolo_labels(downsampled_coco, dest_labels_dir):
    """COCO 형식 데이터에서 YOLO 형식 라벨 파일을 생성합니다."""
    os.makedirs(dest_labels_dir, exist_ok=True)
    
    # 이미지 ID와 크기 매핑
    image_dims = {img['id']: (img['width'], img['height']) for img in downsampled_coco['images']}
    # 이미지 ID와 파일명 매핑
    id_to_filename = {img['id']: img['file_name'] for img in downsampled_coco['images']}
    
    # 이미지별 주석 그룹화
    image_annotations = defaultdict(list)
    for ann in downsampled_coco['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    print(f"YOLO 형식 라벨 생성 중: {dest_labels_dir}")
    for img_id, annotations in tqdm(image_annotations.items()):
        if img_id not in image_dims:
            continue
            
        width, height = image_dims[img_id]
        filename = id_to_filename.get(img_id, "")
        if not filename:
            continue
            
        # 파일명에서 확장자 제거하고 .txt 확장자 사용
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(dest_labels_dir, label_filename)
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # COCO 형식 바운딩 박스를 YOLO 형식으로 변환
                x, y, w, h = ann['bbox']
                
                # 중심점 좌표와 너비/높이를 0-1 범위로 정규화
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_width = w / width
                norm_height = h / height
                
                # 범위 검사 및 조정
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                # 클래스 ID
                class_id = ann['category_id']
                
                # YOLO 형식: <class_id> <x_center> <y_center> <width> <height>
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='데이터셋 다운샘플링 도구')
    parser.add_argument('--coco_json', type=str, required=True, help='COCO 형식 JSON 파일 경로')
    parser.add_argument('--img_dir', type=str, required=True, help='원본 이미지 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='subset', help='출력 디렉토리')
    parser.add_argument('--num_samples', type=int, default=6000, help='선택할 이미지 샘플 수')
    parser.add_argument('--max_per_class', type=int, default=None, help='클래스당 최대 샘플 수')
    parser.add_argument('--min_per_class', type=int, default=20, help='클래스당 최소 샘플 수')
    parser.add_argument('--quality_weight', type=float, default=0.6, help='품질 지표 가중치 (0-1)')
    parser.add_argument('--only_analyze', action='store_true', help='분석만 수행하고 다운샘플링은 수행하지 않음')
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # COCO 데이터셋 분석
    dataset_analysis = analyze_coco_dataset(args.coco_json)
    
    # 클래스 분포 시각화
    visualize_class_distribution(dataset_analysis['class_stats'], args.output_dir)
    
    if args.only_analyze:
        print("분석 완료. --only_analyze 옵션으로 인해 다운샘플링은 수행하지 않습니다.")
        return
    
    # 전략적 다운샘플링 수행
    downsampling_result = stratified_downsampling(
        dataset_analysis,
        target_samples=args.num_samples,
        max_per_class=args.max_per_class,
        min_per_class=args.min_per_class,
        quality_weight=args.quality_weight,
        output_dir=args.output_dir
    )
    
    selected_image_ids = downsampling_result['selected_image_ids']
    downsampled_coco = downsampling_result['downsampled_coco']
    
    # 결과 저장
    # 1. 다운샘플링된 COCO JSON 저장
    coco_output_dir = os.path.join(args.output_dir, 'labels_coco')
    os.makedirs(coco_output_dir, exist_ok=True)
    coco_output_path = os.path.join(coco_output_dir, 'train_downsampled.json')
    
    with open(coco_output_path, 'w', encoding='utf-8') as f:
        json.dump(downsampled_coco, f, ensure_ascii=False, indent=2)
    print(f"다운샘플링된 COCO 데이터 저장됨: {coco_output_path}")
    
    # 2. 선택된 이미지 파일 복사
    img_output_dir = os.path.join(args.output_dir, 'images')
    copy_selected_images(
        selected_image_ids,
        args.img_dir,
        img_output_dir,
        downsampled_coco
    )
    
    # 3. YOLO 형식 라벨 생성
    yolo_output_dir = os.path.join(args.output_dir, 'labels_yolo')
    create_yolo_labels(downsampled_coco, yolo_output_dir)
    
    print(f"다운샘플링 완료. 결과물: {args.output_dir}")
    print(f" - 이미지: {img_output_dir}")
    print(f" - COCO 라벨: {coco_output_dir}")
    print(f" - YOLO 라벨: {yolo_output_dir}")


if __name__ == "__main__":
    main() 