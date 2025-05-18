# CCTV 데이터를 이용한 객체 검출 및 분류

> [!QUOTE] 프로젝트 개요
> 본 저장소는 대전대학교에서 주최한 '지능 IoT 해커톤' 대회 협업을 위한 공간입니다.
> 
> [교통 CCTV 영상 BBOX 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=460)를 이용하여 CCTV 영상 속 객체를 정확하게 추출하는 것이 목적이며, 
> 주요 평가 함수는 mAP를 사용합니다.
>
 >YOLOv8, RF-DETR, D-FINE 등의 다양한 모델과 파라미터 튜닝 등을 통해 최적의 결과를 내는 것이 목적입니다.


## Requirements

- [교통 CCTV 영상 BBOX 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=460)
- 자세한 사항은 environment.yml을 참고하세요.

# 진행 상황

- [x] 데이터셋 전처리 로직 개발
	- [x] YOLO Format
	- [x] COCO Format
- [ ] 모델 학습 로직 개발
	- [x] YOLOv8
	- [ ] RF-DETR
	- [ ] D-FINE