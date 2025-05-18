# CCTV 객체 검출 및 분류 프로젝트

> **대전대학교 '지능 IoT 해커톤' 협업 프로젝트**  
> 본 저장소는 교통 CCTV 영상을 기반으로 객체를 검출하고, 다양한 딥러닝 모델을 비교 및 최적화하기 위한 협업 공간입니다.  
> 주요 평가지표는 **mAP (mean Average Precision)** 입니다.

---

## 데이터셋

- [AIHub 교통 CCTV 영상 BBOX 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=460)

---

## 모델 및 기술 스택

### 사용 모델
- YOLOv8
- RF-DETR
- D-FINE

### 기술 목표
- 객체 검출 성능 극대화
- 다양한 데이터 포맷(YOLO, COCO) 지원
- 모델별 파라미터 튜닝 및 실험 비교

---

## 환경 설정

Python 가상환경 구성은 아래 파일을 참고하세요:

```bash
conda env create -f environment.yml
conda activate cctv-detector
```

