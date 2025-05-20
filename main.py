from src.data_preprocessing import basic_image_preprocess, create_yolo_labels
from src.evaluate import evaluate_detection
from src.inference import generate_prediction
from src.train import train_model
from models.yolo_v8 import YOLOModel
from models.dfine_b import DFineModel
from src.ensemble import ensemble_predict, test_time_augmentation, visualize_detection, run_ensemble, run_tta
import cv2
import os
import argparse
import torch

# cuDNN 벤치마크 모드 함수 추가
def set_cudnn_benchmark(enable=True):
    """cuDNN 벤치마크 모드를 활성화/비활성화 합니다.
    고정된 입력 크기에서 성능 향상을 제공합니다."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = enable
        print(f"cuDNN 벤치마크 모드: {'활성화' if enable else '비활성화'}")


def main():
    """
    CCTV 객체 감지 시스템의 전체 파이프라인을 실행합니다.

    파이프라인 단계:
    1. 데이터 전처리
    2. 라벨 변환: JSON 어노테이션을 YOLO/COCO 형식으로 변환
    3. 모델 학습: 선택한 모델 학습 (YOLOv8 또는 D-FINE)
    4. 추론: 테스트 이미지에서 객체 감지
    5. 평가: 감지 결과 정확도 평가
    """
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="CCTV 객체 감지 시스템")
    parser.add_argument("--model", type=str, choices=["yolo", "dfine"], default="yolo", help="사용할 모델 (yolo, dfine)")
    parser.add_argument("--train", action="store_true", help="모델 학습 실행 여부")
    parser.add_argument("--inference", action="store_true", help="추론 실행 여부")
    parser.add_argument("--train_dir", type=str, default="data/train", help="학습 데이터 디렉토리")
    parser.add_argument("--val_dir", type=str, default="data/val", help="검증 데이터 디렉토리")
    parser.add_argument("--test_dir", type=str, default="data/test", help="테스트 데이터 디렉토리")
    parser.add_argument("--epochs", type=int, default=50, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    parser.add_argument("--img_size", type=int, default=640, help="이미지 크기")
    parser.add_argument("--project", type=str, default="results", help="결과 저장 디렉토리")
    parser.add_argument("--weights", type=str, default=None, help="사전 학습된 가중치 파일 경로")
    parser.add_argument("--train_json", type=str, default=None, help="COCO 형식 학습 주석 파일 (D-FINE용)")
    parser.add_argument("--val_json", type=str, default=None, help="COCO 형식 검증 주석 파일 (D-FINE용)")
    parser.add_argument("--img_root", type=str, default=None, help="이미지 루트 디렉토리 (D-FINE용)")
    
    # 학습 최적화 관련 인자 추가
    parser.add_argument("--use_subset", action="store_true", help="데이터셋의 일부만 사용 (개발/디버깅용)")
    parser.add_argument("--subset_ratio", type=float, default=0.1, help="사용할 데이터셋 비율 (0.1=10%)")
    parser.add_argument("--balanced_sampling", action="store_true", help="클래스 균형을 위한 가중치 샘플링 사용")
    parser.add_argument("--accum_steps", type=int, default=None, help="그래디언트 누적 스텝 수 (None=자동)")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="cuDNN 벤치마크 모드 활성화")
    
    args = parser.parse_args()

    # cuDNN 벤치마크 모드 설정
    if args.cudnn_benchmark:
        set_cudnn_benchmark(True)

    # 모델 선택 및 로드
    model = None
    
    if args.model == "yolo":
        if args.weights:
            print(f"1. YOLOv8 모델 로드: {args.weights}")
            model = YOLOModel()
            model.load(args.weights)
        elif args.train:
            # 3) YOLOv8 모델 학습
            print(f"3. YOLOv8 모델 학습 (에포크: {args.epochs})")
            model = train_model(
                train_dir=args.train_dir, 
                val_dir=args.val_dir, 
                epochs=args.epochs, 
                img_size=args.img_size,
                batch_size=args.batch_size,
                project=args.project
            )
    
    elif args.model == "dfine":
        if args.weights:
            print(f"1. D-FINE 모델 로드: {args.weights}")
            model = DFineModel(num_classes=17, device="cuda" if torch.cuda.is_available() else "cpu")  # 클래스 수에 맞게 조정
            model.load(args.weights)
        elif args.train:
            # D-FINE 모델 학습을 위해 COCO 형식 주석 필요
            if not (args.train_json and args.val_json and args.img_root):
                raise ValueError("D-FINE 학습을 위해서는 --train_json, --val_json, --img_root 인자가 필요합니다.")
            
            print(f"3. D-FINE 모델 학습 (에포크: {args.epochs})")
            model = DFineModel(num_classes=17, project=args.project, device="cuda" if torch.cuda.is_available() else "cpu")  # 클래스 수에 맞게 조정
            model.train(
                train_json=args.train_json,
                val_json=args.val_json,
                img_root=args.img_root,
                epochs=args.epochs,
                batch=args.batch_size,
                log_wandb=False,  # 필요에 따라 활성화
                diffusion_steps=100,  # 디퓨전 스텝 수
                num_workers=4,  # 워커 수 제한 (메모리 사용량 감소)
                amp=True,  # 자동 혼합 정밀도 활성화
                prefetch_factor=2,  # 데이터 미리 가져오기 계수
                monitor_memory=True,  # 메모리 사용량 모니터링
                max_cache_size=500,  # 캐시 크기 제한 (500MB)
                use_subset=args.use_subset,  # 서브셋 사용 여부
                subset_ratio=args.subset_ratio,  # 서브셋 비율
                balanced_sampling=args.balanced_sampling,  # 균등 샘플링 활성화
                accum_steps=args.accum_steps,  # 그래디언트 누적 스텝 수
                aug_cfg={  # 증강 설정
                    "random_resize": [0.7, 1.3],
                    "horizontal_flip": 0.5,
                    "hsv_jitter": 0.2,
                }
            )
            # 학습 후 가중치 저장
            best_weights = f"{args.project}/{model.run_name}/best.pth"
            print(f"최상의 가중치 저장됨: {best_weights}")

    # 추론 및 평가
    if model and args.inference:
        # 4) 테스트 이미지에서 객체 감지 추론 수행
        print("4. 추론 수행 (테스트 데이터)")
        out_label_dir = generate_prediction(model, args.test_dir)
        
        # 5) 평가: 감지 정확도 계산 (정밀도, 재현율, F1 점수)
        print("5. 평가 수행 (IOU 임계값: 0.5)")
        metrics = evaluate_detection(f"{args.test_dir}/labels", out_label_dir, debug=True)
        
        print(
            f"종합 평가 결과 - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
            f" mAP@0.5: {metrics['mAP@0.5']:.4f}"
        )

if __name__ == "__main__":
    main()
