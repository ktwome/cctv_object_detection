from src.data_preprocessing import basic_image_preprocess, create_yolo_labels
from src.evaluate import evaluate_detection
from src.inference import generate_prediction
from src.train import train_model
from models.yolo_v8 import YOLOModel
from src.ensemble import ensemble_predict, test_time_augmentation, visualize_detection, run_ensemble, run_tta
import cv2
import os


def main():
    """
    CCTV 객체 감지 시스템의 전체 파이프라인을 실행합니다.

    파이프라인 단계:
    1. 데이터 전처리
    2. 라벨 변환: JSON 어노테이션을 YOLO 형식으로 변환
    3. 모델 학습: 선택한 모델 학습
    4. 추론: 테스트 이미지에서 객체 감지
    5. 평가: 감지 결과 정확도 평가
    """
    # 이미 학습된 모델 로드
    print("1. 학습된 모델 로드")
    model_path = "results/train/weights/best.pt"
    model = YOLOModel()
    model.load(model_path)
    print(f"모델 로드 완료: {model_path}")

    # 3) 모델 학습
    print("3. 모델 학습 (YOLOv8)")
    model = train_model(train_dir="data/train", val_dir="data/val", epochs=50, project="results")

    # 4) 테스트 이미지에서 객체 감지 추론 수행
    print("4. 추론 수행 (테스트 데이터)")
    out_label_dir = generate_prediction(model, "data/test")
    # out_label_dir = "data/test/labels_pred"

    # 5) 평가: 감지 정확도 계산 (정밀도, 재현율, F1 점수)
    print("5. 평가 수행 (IOU 임계값: 0.5)")
    metrics = evaluate_detection("data/test/labels", out_label_dir, debug=True)

    print(
        f"종합 평가 결과 - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
        f" mAP@0.5: {metrics['mAP@0.5']:.4f}"
    )

if __name__ == "__main__":
    main()
