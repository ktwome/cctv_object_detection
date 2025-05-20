from models.dfine_b import DFineModel
import torch
import os
import argparse
from src.inference import generate_prediction
from src.evaluate import evaluate_detection


def main():
    """
    D-FINE 모델 학습 및 추론 테스트 스크립트
    """
    parser = argparse.ArgumentParser(description="D-FINE 모델 테스트")
    parser.add_argument("--train", action="store_true", help="모델 학습 실행 여부")
    parser.add_argument("--inference", action="store_true", help="추론 실행 여부")
    parser.add_argument("--train_json", type=str, default=None, help="COCO 형식 학습 주석 파일")
    parser.add_argument("--val_json", type=str, default=None, help="COCO 형식 검증 주석 파일")
    parser.add_argument("--img_root", type=str, default=None, help="이미지 루트 디렉토리")
    parser.add_argument("--test_dir", type=str, default="data/test", help="테스트 데이터 디렉토리")
    parser.add_argument("--epochs", type=int, default=50, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--weights", type=str, default=None, help="사전 학습된 가중치 파일 경로")
    parser.add_argument("--project", type=str, default="results/dfine", help="결과 저장 디렉토리")
    args = parser.parse_args()

    # D-FINE 모델 인스턴스 생성
    print("D-FINE-B 모델 초기화 중...")
    model = DFineModel(num_classes=17, project=args.project)  # 클래스 수에 맞게 조정

    # 가중치 로드 또는 학습 수행
    if args.weights:
        print(f"모델 가중치 로드 중: {args.weights}")
        model.load(args.weights)
    elif args.train:
        if not (args.train_json and args.val_json and args.img_root):
            raise ValueError("학습을 위해 --train_json, --val_json, --img_root 인자가 필요합니다.")
        
        print(f"D-FINE 모델 학습 시작 (에포크: {args.epochs}, 배치: {args.batch_size})")
        model.train(
            train_json=args.train_json,
            val_json=args.val_json,
            img_root=args.img_root,
            epochs=args.epochs,
            batch=args.batch_size,
            log_wandb=False,
            diffusion_steps=100,
            aug_cfg={
                "random_resize": [0.7, 1.3],
                "horizontal_flip": 0.5,
                "hsv_jitter": 0.2,
            }
        )
        # 학습된 모델 저장
        best_weights = f"{args.project}/{model.run_name}/best.pth"
        print(f"학습 완료. 최상의 가중치: {best_weights}")
    
    # 추론 및 평가
    if args.inference:
        print(f"D-FINE 모델로 추론 수행 중: {args.test_dir}")
        out_label_dir = generate_prediction(model, args.test_dir)
        
        # 평가 수행
        print("평가 수행 중 (IOU 임계값: 0.5)")
        metrics = evaluate_detection(f"{args.test_dir}/labels", out_label_dir, debug=True)
        
        print(
            f"종합 평가 결과 - Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, "
            f"mAP@0.5: {metrics['mAP@0.5']:.4f}"
        )


if __name__ == "__main__":
    main() 