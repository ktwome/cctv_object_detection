# CCTV 객체 감지 시스템
> CCTV 영상에서 객체를 감지하기 위한 딥러닝 기반 시스템입니다.

## 개요

이 프로젝트는 CCTV 영상에서 자동차, 사람 등의 객체를 감지하기 위한 딥러닝 모델을 학습하고 추론하는 시스템을 제공합니다. YOLOv8 및 D-FINE과 같은 최신 객체 감지 모델을 지원합니다.

## 주요 기능

1. **데이터 전처리**: 이미지 품질 향상 및 라벨 변환
2. **모델 학습**: YOLOv8 또는 D-FINE 모델 학습
3. **객체 감지**: 테스트 이미지에서 객체 감지 수행
4. **성능 평가**: 객체 감지 정확도 평가
5. **앙상블 및 TTA**: 여러 모델 앙상블 및 테스트 시간 증강

## 사용법

### 1. YOLOv8 모델 학습

YOLO 형식의 데이터셋을 사용하여 YOLOv8 모델을 학습합니다.

```bash
python main.py --model yolo --train --train_dir data/train --val_dir data/val --epochs 50 --batch_size 8 --img_size 640 --project results
```

### 2. D-FINE 모델 학습

COCO 형식의 데이터셋을 사용하여 D-FINE 모델을 학습합니다.

```bash
python main.py --model dfine --train --train_json data/train/labels_coco/train.json --val_json data/val/labels_coco/val.json --img_root data/train/images_processed --epochs 60 --batch_size 2 --project results
```

### 3. 학습된 모델로 추론

사전 학습된 가중치를 로드하여 테스트 이미지에서 객체 감지를 수행합니다.

```bash
python main.py --model yolo --inference --weights results/best.pt --test_dir data/test
```

또는 D-FINE 모델:

```bash
python main.py --model dfine --inference --weights results/dfine/best.pt --test_dir data/test
```

### 4. 모델 앙상블

여러 YOLOv8 모델을 결합하여 더 정확한 예측을 생성합니다.

```bash
python main.py --model yolo --inference --weights results/ensemble --test_dir data/test
```

## 매개변수 설명

| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| `--model` | 사용할 모델 (yolo, dfine) | yolo |
| `--train` | 모델 학습 실행 여부 | False |
| `--inference` | 추론 실행 여부 | False |
| `--train_dir` | 학습 데이터 디렉토리 | data/train |
| `--val_dir` | 검증 데이터 디렉토리 | data/val |
| `--test_dir` | 테스트 데이터 디렉토리 | data/test |
| `--epochs` | 학습 에포크 수 | 50 |
| `--batch_size` | 배치 크기 | 8 |
| `--img_size` | 이미지 크기 | 640 |
| `--project` | 결과 저장 디렉토리 | results |
| `--weights` | 사전 학습된 가중치 파일 경로 | None |
| `--train_json` | COCO 형식 학습 주석 파일 (D-FINE용) | None |
| `--val_json` | COCO 형식 검증 주석 파일 (D-FINE용) | None |
| `--img_root` | 이미지 루트 디렉토리 (D-FINE용) | None |

## 모델 지원

### YOLOv8

YOLOv8은 실시간 객체 감지를 위한 빠르고 정확한 모델입니다. 다양한 크기(n, s, m, l, x)를 지원합니다.

### D-FINE

D-FINE은 디퓨전 기반 객체 감지 모델로, 복잡한 장면에서 높은 정확도를 제공합니다. COCO 형식의 데이터셋이 필요합니다.

## 데이터셋 구조 및 요구사항

```
data/
├─test/
│  ├─images/         # 테스트 이미지
│  ├─labels_coco/    # COCO 형식 주석 파일
│  └─labels_json/    # 원본 JSON 형식 주석 파일
├─train/
│  ├─images/         # 원본 학습 이미지
│  ├─images_processed/ # 전처리된 학습 이미지 (그레이스케일, CLAHE 등 적용)
│  ├─labels_coco/    # COCO 형식 주석 파일
│  ├─labels_json/    # 원본 JSON 형식 주석 파일
│  └─labels_yolo/    # YOLO 형식 주석 파일
└─val/
   ├─images/         # 검증 이미지
   ├─labels_coco/    # COCO 형식 주석 파일
   └─labels_json/    # 원본 JSON 형식 주석 파일
```

### YOLOv8 모델

- **디렉토리 구조**: 
  - 이미지: `train_dir/images` 또는 `train_dir/images_processed`(전처리된 경우)
  - 라벨: `train_dir/labels_yolo`
- **라벨 포맷**: YOLO 형식 (.txt)
  - 각 줄: `<class_id> <x_center> <y_center> <width> <height>` (모두 정규화된 좌표 [0-1])

### D-FINE 모델

- **디렉토리 구조**:
  - 이미지: `img_root` (전처리된 이미지 권장: `data/train/images_processed`)
  - 주석: `train_json` (COCO 형식: `data/train/labels_coco/train.json`)
- **라벨 포맷**: COCO 형식 (.json)
  - 주요 섹션: images, annotations, categories
  - 바운딩 박스 형식: [x, y, width, height] (절대 좌표)


  - 하이퍼 파라미터 모델 간 피드백
    -  백 프레셔 기법