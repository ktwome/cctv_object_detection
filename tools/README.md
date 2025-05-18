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
python tools/data_preprocess/yolo_format.py --img_dir data/train/images --json_dir data/train/labels_json --out_dir data/train/labels_yolo --phase train --overwrite
```


### 3. COCO 형식 레이블 변환 (RF-DETR, D-FINE 등에 사용됨)

JSON 형식의 어노테이션을 COCO 호환 포맷으로 변환합니다.

```bash
python tools/data_preprocess/coco_format.py --img_dir data/train/images --json_dir data/train/labels_json --out_dir data/train/labels_coco --phase train --overwrite
```