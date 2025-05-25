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

# 데이터셋 다운샘플링 도구 (dataset_downsampling.py)

다양한 데이터셋 다운샘플링 전략을 제공하는 도구로, 클래스 불균형을 해소하고 품질 좋은 이미지를 선택합니다.

기본 사용법:
```bash
python tools/dataset_downsampling.py --coco_json data/train/labels_coco/train.json --img_dir data/train/images --output_dir balanced_subset --num_samples 6000
```

주요 기능:
1. 클래스별 균형 있는 다운샘플링 (기본)
2. 최소 클래스 기준 균등 샘플링 (--balance_by_min_class)
3. 특정 클래스 제외 (--exclude_classes)
4. 품질 기준 선별 (바운딩 박스 크기, 위치, 종횡비, 가시성 고려)
5. 자동 train/val/test 분할 (--split_data)
6. 분할 후 클래스 분포 시각화
7. 데이터셋 검증 기능 (--verify_coco)

특정 클래스를 제외하고 최소 클래스 기준으로 샘플링하는 예시:
```bash
python tools/dataset_downsampling.py --coco_json data/train/labels_coco/train.json --img_dir data/train/images --output_dir balanced_subset --exclude_classes "통학버스" "구급차" "소방차" "어린이" "자전거 / 기타 전동 이동체" "삼각대" --balance_by_min_class --split_data
```

데이터셋 검증 및 자동 수정 예시:
```bash
python tools/dataset_downsampling.py --coco_json data/train/labels_coco/train.json --img_dir data/train/images --verify_coco
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
| `--visualize_top` | 클래스별 시각화할 최고 품질 객체 수 | 5 |
| `--exclude_classes` | 제외할 클래스 이름 목록 | None |
| `--balance_by_min_class` | 포함된 클래스 중 최소 샘플 수를 기준으로 균등하게 샘플링 | False |
| `--split_data` | 데이터를 학습/검증/테스트 세트로 분리 | False |
| `--train_ratio` | 학습 데이터 비율 | 0.7 |
| `--val_ratio` | 검증 데이터 비율 | 0.2 |
| `--test_ratio` | 테스트 데이터 비율 | 0.1 |
| `--random_seed` | 랜덤 시드 (데이터 분할 재현성) | 42 |

## 출력 디렉토리 구조

### 기본 다운샘플링 출력

```
subset/
├── class_distribution.png     # 클래스 분포 시각화
├── images/                    # 다운샘플링된 이미지
├── labels_coco/               # COCO 형식 라벨
│   └── train_downsampled.json
├── labels_yolo/               # YOLO 형식 라벨
└── top_quality_visualization/ # 클래스별 최고 품질 객체 시각화
```

### 분할 모드 출력 (--split_data 사용 시)

```
dataset_split/
├── class_distribution.png     # 원본 클래스 분포 시각화
├── split_distribution.png     # 분할 후 클래스 분포 시각화
├── train/                     # 학습 데이터셋
│   ├── images/                # 학습 이미지
│   ├── labels_coco/           # COCO 형식 라벨
│   │   └── train.json
│   └── labels_yolo/           # YOLO 형식 라벨
├── val/                       # 검증 데이터셋
│   ├── images/                # 검증 이미지
│   ├── labels_coco/           # COCO 형식 라벨
│   │   └── val.json
│   └── labels_yolo/           # YOLO 형식 라벨
├── test/                      # 테스트 데이터셋
│   ├── images/                # 테스트 이미지
│   ├── labels_coco/           # COCO 형식 라벨
│   │   └── test.json
│   └── labels_yolo/           # YOLO 형식 라벨
└── top_quality_visualization/ # 클래스별 최고 품질 객체 시각화
```

## 다운샘플링 방법

이 스크립트는 단순한 무작위 샘플링이 아닌 다음과 같은 전략적 다운샘플링을 수행합니다:

1. **클래스 균형**: 모든 클래스가 적절히 대표될 수 있도록 클래스별 할당량을 계산합니다.
2. **품질 평가**: 각 이미지에 대해 다음 요소를 고려한 품질 점수를 계산합니다:
   - 객체 크기 (적절한 크기의 객체 선호)
   - 객체 위치 (이미지 중앙에 위치한 객체 선호)
   - 객체 가시성 (잘리지 않은 완전한 객체 선호)
   - 객체 종횡비 (균형 잡힌 종횡비 선호)
   - 다양성 (여러 클래스를 포함하는 이미지 선호)
   - 혼잡도 (너무 많은 객체가 있는 이미지에 패널티)
3. **점수 기반 선택**: 각 클래스별로 품질 점수가 높은 상위 이미지를 선택합니다.
4. **균등 샘플링 옵션**: `--balance_by_min_class` 옵션을 사용하면 포함된 클래스 중 가장 적은 샘플 수를 가진 클래스를 기준으로 모든 클래스에서 동일한 수의 샘플을 추출합니다.
5. **데이터셋 분할**: `--split_data` 옵션을 사용하면 다운샘플링된 데이터를 train, val, test 세트로 나누고 각 세트의 클래스 분포를 시각화합니다.

# 라바콘 시각화 도구 (visualize_rabacon.py)

COCO 형식 데이터셋에서 라바콘 클래스가 포함된 이미지만 추출하여 바운딩 박스를 시각화하는 도구입니다.

## 기능

1. **라바콘 객체 추출**: COCO 형식 데이터셋에서 라바콘 클래스를 가진 모든 객체 추출
2. **객체 크기별 샘플링**: 크기별(큰/중간/작은)로 고르게 샘플 선택
3. **바운딩 박스 시각화**: 라바콘 객체의 바운딩 박스와 면적 표시
4. **고품질 이미지 생성**: 라바콘 시각화 결과를 고품질 이미지로 저장

## 사용법

### 1. 라바콘 50개 샘플 시각화 (기본값)

```bash
python tools/visualize_rabacon.py --coco_json data/train/labels_coco/train.json --img_dir data/train/images --output_dir rabacon_visualization
```

### 2. 더 많은 라바콘 샘플 시각화 (예: 1000개)

```bash
python tools/visualize_rabacon.py --coco_json data/train/labels_coco/train.json --img_dir data/train/images --output_dir rabacon_visualization --num_samples 1000
```

## 주요 매개변수

| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| `--coco_json` | COCO 형식 JSON 파일 경로 | (필수) |
| `--img_dir` | 원본 이미지 디렉토리 경로 | (필수) |
| `--output_dir` | 시각화 결과 저장 디렉토리 | rabacon_visualization |
| `--num_samples` | 시각화할 샘플 수 | 50 |

## 출력 디렉토리 구조

```
rabacon_visualization/
├── rabacon_1_area1234.jpg   # 라바콘 객체 시각화 (면적 포함)
├── rabacon_2_area987.jpg
├── ...
└── rabacon_N_area456.jpg
```

## 시각화 방법

이 스크립트는 라바콘 객체를 다음과 같은 방식으로 시각화합니다:

1. **크기별 샘플링**: 라바콘 객체를 크기(면적)에 따라 큰/중간/작은 그룹으로 나눕니다.
2. **균등한 선택**: 각 그룹에서 균등하게 샘플을 선택하여 다양한 크기의 라바콘 객체를 볼 수 있도록 합니다.
3. **바운딩 박스 표시**: 빨간색 사각형으로 라바콘 객체의 바운딩 박스를 표시합니다.
4. **면적 정보 표시**: 각 라바콘 객체의 면적을 텍스트로 함께 표시합니다.

### 5. Windows 환경에서의 데이터셋 사용 주의사항

Windows 환경에서는 파일 경로 표기 방식(역슬래시 \)으로 인해 데이터셋에 문제가 발생할 수 있습니다. dataset_downsampling.py 도구는 다음과 같은 방법으로 이 문제를 해결합니다:

1. 파일 경로를 Path 객체로 처리 (pathlib 사용)
2. 파일명과 디렉토리 경로를 별도로 처리
3. 파일 이름만으로도 찾을 수 있도록 파일명 검색
4. COCO JSON 파일의 file_name 필드 자동 정규화

데이터셋에서 파일 경로 문제가 발생하면 --verify_coco 옵션을 사용하여 검증 및 수정을 수행하세요:
```bash
python tools/dataset_downsampling.py --coco_json data/train/labels_coco/train.json --img_dir data/train/images --verify_coco
```

### 6. 진행 상황 시각화 (tqdm)

모든 데이터 처리 도구는 tqdm 라이브러리를 사용하여 진행 상황을 시각적으로 표시합니다. 설치되지 않은 경우 다음 명령어로 설치할 수 있습니다:

```bash
pip install tqdm
```

tqdm이 설치되어 있지 않아도 도구는 작동하지만, 진행 상황이 텍스트로만 표시됩니다.
