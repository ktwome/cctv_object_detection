# Tools
> 최초 1회만 실행되는 도구들을 모아둔 디렉토리입니다.

# 사용법

## 데이터셋 추출기

원본 데이터셋 압축파일을 프로젝트에서 요구되는 사항을 충족하도록 해제합니다.

1. https://aihub.or.kr/aihubdata/data/dwld.do?currMenu=115&topMenu=100 에서 데이터를 다운로드 받는다.
2. 다운로드 받은 데이터를 압축 해제한다.
3. 압축 해제된 압축의 상단 디렉토리에 extractor.py를 넣는다.
4. 터미널에서 extractor.py를 실행한다.

```bash
python extractor.py
```


## 데이터 전처리 도구들
> *참고* : 모든 데이터 전처리 도구의 명령어는 프로젝트의 루트 디렉토리에서 실행해야 합니다.

### 1. 기초 이미지 전처리 (그레이스케일, CLAHE)

이미지 가독성을 높이기 위한 기본 전처리를 수행합니다.

- 그레이스케일 변환
- 대비 제한 적응형 히스토그램 평활화 (CLAHE, Contrast Limited Adaptive Histogram Equalization) 적용

```bash
python tools/data_preprocess/basic_image_preprocess.py --input_dir data/train/images --output_dir data/train/images_processed --use_gray --use_clahe --overwrite
```

### 2. YOLO 형식 레이블 변환 (YOLOv8에 사용됨)

JSON 형식의 어노테이션을 YOLOv8 호환 포맷으로 변환합니다.

```bash
python tools/data_preprocess/yolo_format.py --base_dir ./data --all
```


### 3. COCO 형식 레이블 변환 (RF-DETR, D-FINE 등에 사용됨)

JSON 형식의 어노테이션을 COCO 호환 포맷으로 변환합니다.

```bash
python tools/data_preprocess/coco_format.py --base_dir ./data --all
```

# 데이터셋 다운샘플링 도구

극심한 클래스 불균형(최대 13만개 : 최소 300개) 문제를 해결하기 위한 도구입니다.

## 기능

1. **데이터셋 분석**: COCO 형식 JSON 파일을 분석하여 클래스별 분포 통계 제공
2. **클래스 분포 시각화**: 클래스별 객체 수 분포 시각화 그래프 생성
3. **전략적 다운샘플링**: 단순 무작위 샘플링이 아닌 클래스 균형과 데이터 품질을 고려한 다운샘플링
4. **COCO/YOLO 형식 변환**: 다운샘플링된 데이터를 다양한 형식으로 변환하여 저장

## 설치

필요한 패키지를 설치하려면 다음 명령을 실행하세요:

```bash
python tools/setup_downsampling.py
```

## 사용법

### 1. 데이터셋 분석만 수행

```bash
python tools/dataset_downsampling.py --coco_json data/train/labels_coco/train.json --img_dir data/train/images --only_analyze
```

### 2. 다운샘플링 수행 (6000개 샘플 추출)

```bash
python tools/dataset_downsampling.py --coco_json data/train/labels_coco/train.json --img_dir data/train/images --output_dir subset --num_samples 6000
```

### 3. 클래스별 최소/최대 샘플 수 지정

```bash
python tools/dataset_downsampling.py --coco_json data/train/labels_coco/train.json --img_dir data/train/images --output_dir subset --num_samples 6000 --min_per_class 20 --max_per_class 500
```

## 주요 매개변수

| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| `--coco_json` | COCO 형식 JSON 파일 경로 | (필수) |
| `--img_dir` | 원본 이미지 디렉토리 경로 | (필수) |
| `--output_dir` | 출력 디렉토리 | subset |
| `--num_samples` | 선택할 이미지 샘플 수 | 6000 |
| `--max_per_class` | 클래스당 최대 샘플 수 | None |
| `--min_per_class` | 클래스당 최소 샘플 수 | 20 |
| `--quality_weight` | 품질 지표 가중치 (0-1) | 0.6 |
| `--only_analyze` | 분석만 수행하고 다운샘플링은 수행하지 않음 | False |

## 출력 디렉토리 구조

```
subset/
├── class_distribution.png   # 클래스 분포 시각화
├── images/                  # 다운샘플링된 이미지
├── labels_coco/             # COCO 형식 라벨
│   └── train_downsampled.json
└── labels_yolo/             # YOLO 형식 라벨
```

## 다운샘플링 방법

이 스크립트는 단순한 무작위 샘플링이 아닌 다음과 같은 전략적 다운샘플링을 수행합니다:

1. **클래스 균형**: 모든 클래스가 적절히 대표될 수 있도록 클래스별 할당량을 계산합니다.
2. **품질 평가**: 각 이미지에 대해 다음 요소를 고려한 품질 점수를 계산합니다:
   - 객체 크기 (큰 객체가 더 명확히 보임)
   - 다양성 (여러 클래스를 포함하는 이미지 선호)
   - 혼잡도 (너무 많은 객체가 있는 이미지에 패널티)
3. **점수 기반 선택**: 각 클래스별로 품질 점수가 높은 상위 이미지를 선택합니다.