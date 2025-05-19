# CCTV 데이터를 이용한 객체 검출 및 분류

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

## 1. 전처리

- [x] 1. 데이터셋 전처리 로직 개발
	- [x] YOLO Format
	- [x] COCO Format
---
## 2. 서브셋 탐색 (하이퍼파라미터 튜닝)

- [ ] 데이터셋 추출
	- 전체 데이터셋의 10%
	- 클래스별 균등 추출
	- 시드 고정을 통한 재현성 확보
	- 결과를 subset 디렉토리 내 저장
	- Train : Val = 9 : 1 분리
		- json으로 파일 목록 및 시드 기록

- [ ] 분포 검증
	- 전체 vs 서브셋 클래스 히스토그램 출력
	- 편차 +-3% 이상시 경고
	- matplotlib 그래프 저장

- [ ] HyperBand 스윕 구현

---
## 3. 파일럿 트레이닝

- [ ] 3. 예비 학습 (Pilot Training)
	- [ ] 예비 학습 환경 세팅

---
- [ ] 모델 학습 로직 개발
	- [x] YOLOv8 (?)
	- [ ] RF-DETR-Swin-T
	- [ ] YOLOv9-E
	- [ ] D-FINE-B