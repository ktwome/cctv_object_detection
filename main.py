from src.data_preprocessing import basic_image_preprocess, create_yolo_labels, process_images_in_directory, apply_custom_preprocessing
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
import yaml
from src.hyperparameter_tuning import run_hyperparameter_tuning
import shutil
import re # re 모듈 추가
import glob

# cuDNN 벤치마크 모드 함수 추가
def set_cudnn_benchmark(enable=True):
    """cuDNN 벤치마크 모드를 활성화/비활성화 합니다.
    고정된 입력 크기에서 성능 향상을 제공합니다."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = enable
        print(f"cuDNN 벤치마크 모드: {'활성화' if enable else '비활성화'}")


def find_resume_checkpoint(resume_path):
    # 1. 명시적으로 입력한 경로가 존재하면 그대로 사용
    if resume_path and os.path.exists(resume_path):
        print(f"[INFO][find_resume_checkpoint] 명시적으로 지정된 resume 경로 사용: {resume_path}")
        return resume_path
    # 2. runs/detect 하위 폴더에서 last.pt 우선 탐색
    last_pts = glob.glob("runs/detect/**/last.pt", recursive=True)
    print(f"[DEBUG][find_resume_checkpoint] runs/detect/**/last.pt 탐색 결과: {last_pts}")
    if last_pts:
        # 가장 최근 수정된 last.pt 선택
        last_pts.sort(key=os.path.getmtime, reverse=True)
        print(f"[INFO][find_resume_checkpoint] runs/detect 하위에서 last.pt 자동 선택: {last_pts[0]}")
        return last_pts[0]
    # 3. 마지막 수단: None 반환
    print("[WARNING][find_resume_checkpoint] resume 체크포인트를 찾지 못했습니다. 기본 모델로 시작합니다.")
    return None


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
    parser.add_argument("--imgsz", type=int, default=640, help="이미지 크기")
    parser.add_argument("--project", type=str, default="results", help="결과 저장 디렉토리")
    parser.add_argument("--weights", type=str, default=None, help="사전 학습된 가중치 파일 경로")
    parser.add_argument("--device", type=str, default="auto", help="학습 및 추론에 사용할 장치 (예: 'cpu', 'cuda', '0', 'auto')")
    parser.add_argument("--workers", type=int, default=8, help="데이터 로딩에 사용할 워커 수")
    parser.add_argument("--train_json", type=str, default=None, help="COCO 형식 학습 주석 파일 (D-FINE용)")
    parser.add_argument("--val_json", type=str, default=None, help="COCO 형식 검증 주석 파일 (D-FINE용)")
    parser.add_argument("--img_root", type=str, default=None, help="이미지 루트 디렉토리 (D-FINE용)")
    parser.add_argument("--val_img_root", type=str, default=None, help="검증 이미지 루트 디렉토리 (D-FINE용)")
    
    # 학습 최적화 관련 인자 추가
    parser.add_argument("--use_subset", action="store_true", help="데이터셋의 일부만 사용 (개발/디버깅용)")
    parser.add_argument("--subset_ratio", type=float, default=0.1, help="사용할 데이터셋 비율 (0.1=10%)")
    parser.add_argument("--balanced_sampling", action="store_true", help="클래스 균형을 위한 가중치 샘플링 사용")
    parser.add_argument("--accum_steps", type=int, default=None, help="그래디언트 누적 스텝 수 (None=자동)")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="cuDNN 벤치마크 모드 활성화")
    
    # 하이퍼파라미터 튜닝 관련 인자 추가
    parser.add_argument("--tune_hyperparameters", action="store_true", help="Optuna를 사용하여 하이퍼파라미터 튜닝 실행")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna 튜닝 시도 횟수")
    parser.add_argument("--tuning_epochs", type=int, default=None, help="튜닝 시 사용할 최대 에포크 (None이면 args.epochs 사용)")
    parser.add_argument("--yolo_model_size_for_tuning", type=str, default="s", choices=['n', 's', 'm', 'l', 'x'], help="튜닝 시 사용할 YOLO 모델 크기")
    
    # 추가된 인자
    parser.add_argument("--resume", type=str, default=None, help="이전 학습 결과(체크포인트)에서 이어서 학습할 경로 (예: runs/detect/train/weights/last.pt)")
    
    args = parser.parse_args()

    # cuDNN 벤치마크 모드 설정
    if args.cudnn_benchmark:
        set_cudnn_benchmark(True)

    # 장치 설정
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"사용할 장치: {device}")

    # 모델 선택 및 로드
    model = None
    
    if args.model == "yolo":
        # 학습 데이터 경로 설정 (전처리 여부 확인)
        train_images_dir_to_use = os.path.join(args.train_dir, "images")
        # train_labels_dir_to_use = os.path.join(args.train_dir, "labels") # 라벨 경로는 data.yaml에 직접 명시되지 않음.
        val_images_dir_to_use = os.path.join(args.val_dir, "images")
        # val_labels_dir_to_use = os.path.join(args.val_dir, "labels")

        # 전처리된 이미지 디렉토리 경로
        processed_train_images_dir = os.path.join(args.train_dir, "images_processed")
        processed_val_images_dir = os.path.join(args.val_dir, "images_processed")

        # data.yaml 파일 경로 (학습 또는 튜닝 시 생성/사용)
        custom_yaml_path = os.path.join(args.train_dir, "custom_data.yaml")

        # 학습 또는 튜닝 모드 모두에서 데이터 전처리 및 data.yaml 생성은 공통적으로 필요할 수 있음
        # (튜닝 시에도 전처리된 데이터를 사용하고, data.yaml을 참조하므로)
        if args.train or args.tune_hyperparameters:
            print("\n[INFO] --- 학습 또는 튜닝을 위한 데이터 준비 시작 ---")
            
            # --- 학습 데이터 전처리 및 라벨 처리 ---
            print(f"[DEBUG] 원본 학습 이미지 디렉토리: {os.path.join(args.train_dir, 'images')}")
            if not os.path.exists(processed_train_images_dir):
                print(f"[INFO] 전처리된 학습 이미지 디렉토리 ('{processed_train_images_dir}')가 없습니다. 새로 생성합니다.")
                if not os.path.exists(os.path.join(args.train_dir, 'images')):
                    raise FileNotFoundError(f"원본 학습 이미지 디렉토리를 찾을 수 없습니다: {os.path.join(args.train_dir, 'images')}")
                
                print(f"[INFO] '{os.path.join(args.train_dir, 'images')}'의 이미지를 전처리하여 '{processed_train_images_dir}'에 저장 시작...")
                process_images_in_directory(os.path.join(args.train_dir, 'images'), processed_train_images_dir, 
                                            overwrite=False, use_clahe=True, use_noise_reduction=True)
                print(f"[INFO] 학습 이미지 전처리 완료. 저장된 경로: {processed_train_images_dir}")
            else:
                print(f"[INFO] 이미 전처리된 학습 이미지 디렉토리 ('{processed_train_images_dir}')가 존재합니다. 전처리 생략.")
            
            if os.path.exists(processed_train_images_dir):
                print(f"[DEBUG] 전처리된 학습 이미지 파일 수: {len(os.listdir(processed_train_images_dir))}")

                original_train_labels_dir_yolo = os.path.join(args.train_dir, "labels_yolo")
                original_train_labels_dir_labels = os.path.join(args.train_dir, "labels")
                actual_original_train_labels_dir = None

                print(f"[DEBUG] 원본 학습 라벨 (labels_yolo) 경로 확인 중: {original_train_labels_dir_yolo}")
                if os.path.exists(original_train_labels_dir_yolo):
                    actual_original_train_labels_dir = original_train_labels_dir_yolo
                    print(f"[DEBUG] '{original_train_labels_dir_yolo}' 사용. 파일 수: {len(os.listdir(original_train_labels_dir_yolo))}")
                else:
                    print(f"[DEBUG] '{original_train_labels_dir_yolo}' 없음. 원본 학습 라벨 (labels) 경로 확인 중: {original_train_labels_dir_labels}")
                    if os.path.exists(original_train_labels_dir_labels):
                        actual_original_train_labels_dir = original_train_labels_dir_labels
                        print(f"[DEBUG] '{original_train_labels_dir_labels}' 사용. 파일 수: {len(os.listdir(original_train_labels_dir_labels))}")
                    else:
                        print(f"[ERROR] 원본 학습 라벨 디렉토리를 찾을 수 없습니다: '{original_train_labels_dir_yolo}' 또는 '{original_train_labels_dir_labels}'")
                
                target_train_labels_dir = os.path.join(processed_train_images_dir, "..", "labels")
                os.makedirs(target_train_labels_dir, exist_ok=True)
                print(f"[DEBUG] 최종 학습 라벨 저장 경로: {target_train_labels_dir}")

                if actual_original_train_labels_dir:
                    print(f"[INFO] '{actual_original_train_labels_dir}'의 라벨을 '{target_train_labels_dir}'로 복사 및 이름 변경 시작...")
                    copied_labels_count = 0
                    not_found_labels_count = 0
                    for processed_img_filename_with_ext in os.listdir(processed_train_images_dir):
                        processed_img_basename, img_ext = os.path.splitext(processed_img_filename_with_ext)
                        if img_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                            original_label_basename = processed_img_basename
                            # print(f"[TRACE] 처리 중인 이미지: {processed_img_filename_with_ext}, 베이스명: {processed_img_basename}")
                            match = re.match(r"^@aug_[a-zA-Z0-9]+_(.*)", processed_img_basename)
                            if match:
                                original_label_basename = match.group(1)
                                # print(f"[TRACE] 접두사 제거 후 원본 라벨 베이스명: {original_label_basename}")
                            
                            original_label_filename_txt = original_label_basename + ".txt"
                            original_label_path = os.path.join(actual_original_train_labels_dir, original_label_filename_txt)
                            
                            target_label_filename_txt = processed_img_basename + ".txt" # 최종 라벨 파일명은 이미지명과 동일하게 (접두사 포함)
                            target_label_path = os.path.join(target_train_labels_dir, target_label_filename_txt)

                            if os.path.exists(original_label_path):
                                shutil.copy2(original_label_path, target_label_path)
                                copied_labels_count += 1
                            else:
                                # print(f"[TRACE] 원본 라벨 파일 없음: {original_label_path}")
                                not_found_labels_count +=1
                    print(f"[INFO] 학습 라벨 복사 완료. 복사된 파일: {copied_labels_count}개, 찾지 못한 파일: {not_found_labels_count}개")
                    if not_found_labels_count > 0:
                        print(f"[WARNING] {not_found_labels_count}개의 원본 학습 라벨을 찾지 못했습니다. 파일명 접두사 규칙, 원본 라벨 폴더명('labels_yolo' 또는 'labels'), 또는 원본 라벨 파일 존재 여부를 확인하세요.")
            
            train_images_dir_to_use = processed_train_images_dir
            print(f"[INFO] 사용할 최종 학습 이미지 디렉토리: {train_images_dir_to_use}")

            # --- 검증 데이터 전처리 및 라벨 처리 (학습 데이터와 유사하게 로그 추가) ---
            print(f"[DEBUG] 원본 검증 이미지 디렉토리: {os.path.join(args.val_dir, 'images')}")
            if not os.path.exists(processed_val_images_dir):
                print(f"[INFO] 전처리된 검증 이미지 디렉토리 ('{processed_val_images_dir}')가 없습니다. 새로 생성합니다.")
                if not os.path.exists(os.path.join(args.val_dir, 'images')):
                    print(f"[WARNING] 원본 검증 이미지 디렉토리를 찾을 수 없습니다: {os.path.join(args.val_dir, 'images')}. 원본 경로를 사용 시도합니다.")
                    val_images_dir_to_use = os.path.join(args.val_dir, 'images') 
                else:
                    print(f"[INFO] '{os.path.join(args.val_dir, 'images')}'의 이미지를 전처리하여 '{processed_val_images_dir}'에 저장 시작...")
                    process_images_in_directory(os.path.join(args.val_dir, 'images'), processed_val_images_dir, 
                                                overwrite=False, use_clahe=True, use_noise_reduction=True)
                    print(f"[INFO] 검증 이미지 전처리 완료. 저장된 경로: {processed_val_images_dir}")
                    val_images_dir_to_use = processed_val_images_dir # 전처리된 경로 사용
            else:
                print(f"[INFO] 이미 전처리된 검증 이미지 디렉토리 ('{processed_val_images_dir}')가 존재합니다. 전처리 생략.")
                val_images_dir_to_use = processed_val_images_dir # 이미 있으면 해당 경로 사용

            if os.path.exists(val_images_dir_to_use) and "images_processed" in val_images_dir_to_use : # 전처리된 이미지를 사용하는 경우에만 라벨 복사
                print(f"[DEBUG] 전처리된 검증 이미지 파일 수: {len(os.listdir(val_images_dir_to_use)) if os.path.exists(val_images_dir_to_use) else 0}")
                original_val_labels_dir_yolo = os.path.join(args.val_dir, "labels_yolo")
                original_val_labels_dir_labels = os.path.join(args.val_dir, "labels")
                actual_original_val_labels_dir = None

                print(f"[DEBUG] 원본 검증 라벨 (labels_yolo) 경로 확인 중: {original_val_labels_dir_yolo}")
                if os.path.exists(original_val_labels_dir_yolo):
                    actual_original_val_labels_dir = original_val_labels_dir_yolo
                    print(f"[DEBUG] '{original_val_labels_dir_yolo}' 사용. 파일 수: {len(os.listdir(original_val_labels_dir_yolo))}")
                else:
                    print(f"[DEBUG] '{original_val_labels_dir_yolo}' 없음. 원본 검증 라벨 (labels) 경로 확인 중: {original_val_labels_dir_labels}")
                    if os.path.exists(original_val_labels_dir_labels):
                        actual_original_val_labels_dir = original_val_labels_dir_labels
                        print(f"[DEBUG] '{original_val_labels_dir_labels}' 사용. 파일 수: {len(os.listdir(original_val_labels_dir_labels))}")
                    else:
                        print(f"[ERROR] 원본 검증 라벨 디렉토리를 찾을 수 없습니다: '{original_val_labels_dir_yolo}' 또는 '{original_val_labels_dir_labels}'")

                target_val_labels_dir = os.path.join(val_images_dir_to_use, "..", "labels") # val_images_dir_to_use가 processed_val_images_dir일때
                os.makedirs(target_val_labels_dir, exist_ok=True)
                print(f"[DEBUG] 최종 검증 라벨 저장 경로: {target_val_labels_dir}")

                if actual_original_val_labels_dir:
                    print(f"[INFO] '{actual_original_val_labels_dir}'의 라벨을 '{target_val_labels_dir}'로 복사 및 이름 변경 시작...")
                    copied_labels_count = 0
                    not_found_labels_count = 0
                    for processed_img_filename_with_ext in os.listdir(val_images_dir_to_use): # val_images_dir_to_use (processed 된 폴더)
                        processed_img_basename, img_ext = os.path.splitext(processed_img_filename_with_ext)
                        if img_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                            original_label_basename = processed_img_basename
                            match = re.match(r"^@aug_[a-zA-Z0-9]+_(.*)", processed_img_basename)
                            if match:
                                original_label_basename = match.group(1)
                            
                            original_label_filename_txt = original_label_basename + ".txt"
                            original_label_path = os.path.join(actual_original_val_labels_dir, original_label_filename_txt)
                            
                            target_label_filename_txt = processed_img_basename + ".txt"
                            target_label_path = os.path.join(target_val_labels_dir, target_label_filename_txt)

                            if os.path.exists(original_label_path):
                                shutil.copy2(original_label_path, target_label_path)
                                copied_labels_count +=1
                            else:
                                not_found_labels_count +=1
                    print(f"[INFO] 검증 라벨 복사 완료. 복사된 파일: {copied_labels_count}개, 찾지 못한 파일: {not_found_labels_count}개")
                    if not_found_labels_count > 0:
                        print(f"[WARNING] {not_found_labels_count}개의 원본 검증 라벨을 찾지 못했습니다.")
            else: # 원본 val 이미지 경로를 그대로 사용하는 경우 (전처리 안함/못함)
                 print(f"[INFO] 검증 이미지는 원본 경로 '{val_images_dir_to_use}'를 사용합니다. 라벨은 해당 경로 기준으로 YOLO가 찾습니다 (예: '{os.path.join(val_images_dir_to_use, '../labels')}')")


            print(f"[INFO] 사용할 최종 검증 이미지 디렉토리: {val_images_dir_to_use}")
            
            # data.yaml 생성
            temp_yolo_model = YOLOModel(device=device) 
            class_names = temp_yolo_model.get_class_names() 
            if not class_names:
                class_names = ["경차/세단", "SUV/승합차", "트럭", "버스(소형, 대형)", "통학버스(소형,대형)", "경찰차", "구급차", "소방차", "견인차", "기타 특장차", "성인", "어린이", "오토바이", "자전거 / 기타 전동 이동체", "라바콘", "삼각대", "기타"]
                print("[WARNING] YOLOModel에서 클래스 이름을 가져오지 못해 임시 클래스 목록을 사용합니다.")

            # 경로 구분자를 '/'로 통일
            train_path_for_yaml = os.path.abspath(train_images_dir_to_use).replace('\\', '/')
            val_path_for_yaml = os.path.abspath(val_images_dir_to_use).replace('\\', '/')

            # 라벨 경로 지정 (labels_yolo 폴더 우선, 없으면 labels 폴더)
            train_labels_dir = None
            if os.path.exists(os.path.join(args.train_dir, "labels_yolo")):
                train_labels_dir = os.path.abspath(os.path.join(args.train_dir, "labels_yolo")).replace('\\', '/')
            elif os.path.exists(os.path.join(args.train_dir, "labels")):
                train_labels_dir = os.path.abspath(os.path.join(args.train_dir, "labels")).replace('\\', '/')
            val_labels_dir = None
            if os.path.exists(os.path.join(args.val_dir, "labels_yolo")):
                val_labels_dir = os.path.abspath(os.path.join(args.val_dir, "labels_yolo")).replace('\\', '/')
            elif os.path.exists(os.path.join(args.val_dir, "labels")):
                val_labels_dir = os.path.abspath(os.path.join(args.val_dir, "labels")).replace('\\', '/')

            yolo_data_yaml_content = {
                "train": train_path_for_yaml,
                "val": val_path_for_yaml,
                "names": {i: name for i, name in enumerate(class_names)},
                "nc": len(class_names)
            }
            # train_labels, val_labels 항목 추가 (존재할 때만)
            if train_labels_dir:
                yolo_data_yaml_content["train_labels"] = train_labels_dir
            if val_labels_dir:
                yolo_data_yaml_content["val_labels"] = val_labels_dir

            with open(custom_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(yolo_data_yaml_content, f, default_flow_style=False, allow_unicode=True)
            
            print(f"[INFO] 커스텀 data.yaml 생성: {custom_yaml_path}")
            print(f"  ㄴ train: {train_path_for_yaml}")
            print(f"  ㄴ val: {val_path_for_yaml}")
            print(f"  ㄴ 최종 학습 라벨 예상 경로: {os.path.abspath(target_train_labels_dir).replace('\\\\','/')}")
            if "images_processed" in val_images_dir_to_use : # 전처리된 이미지를 사용하는 경우에만 해당 경로 출력
                 print(f"  ㄴ 최종 검증 라벨 예상 경로: {os.path.abspath(target_val_labels_dir).replace('\\\\','/')}")
            else: # 원본 검증 이미지를 사용하는 경우
                 print(f"  ㄴ 최종 검증 라벨 예상 경로: {os.path.abspath(os.path.join(val_images_dir_to_use, '../labels')).replace('\\\\','/')}")

            print("[INFO] --- 데이터 준비 완료 ---")

            # --- images_processed 폴더에 라벨(.txt) 파일 복사 ---
            def copy_labels_to_images_processed(images_dir, labels_dir):
                if not os.path.exists(labels_dir):
                    print(f"[경고] 라벨 디렉토리 없음: {labels_dir}")
                    return
                count = 0
                for img_file in os.listdir(images_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        base = os.path.splitext(img_file)[0]
                        label_file = base + ".txt"
                        src = os.path.join(labels_dir, label_file)
                        dst = os.path.join(images_dir, label_file)
                        if os.path.exists(src):
                            if not os.path.exists(dst):
                                shutil.copy2(src, dst)
                                count += 1
                print(f"[INFO] images_processed 폴더에 라벨 복사 완료: {count}개")

            # 학습용 라벨 복사
            train_labels_yolo = os.path.join(args.train_dir, "labels_yolo")
            train_labels = os.path.join(args.train_dir, "labels")
            if os.path.exists(processed_train_images_dir):
                if os.path.exists(train_labels_yolo):
                    copy_labels_to_images_processed(processed_train_images_dir, train_labels_yolo)
                elif os.path.exists(train_labels):
                    copy_labels_to_images_processed(processed_train_images_dir, train_labels)

            # 검증용 라벨 복사
            val_labels_yolo = os.path.join(args.val_dir, "labels_yolo")
            val_labels = os.path.join(args.val_dir, "labels")
            if os.path.exists(processed_val_images_dir):
                if os.path.exists(val_labels_yolo):
                    copy_labels_to_images_processed(processed_val_images_dir, val_labels_yolo)
                elif os.path.exists(val_labels):
                    copy_labels_to_images_processed(processed_val_images_dir, val_labels)

        if args.tune_hyperparameters:
            print("\n--- 하이퍼파라미터 튜닝 시작 ---")
            if args.model != "yolo":
                print("경고: 하이퍼파라미터 튜닝은 현재 YOLO 모델에 대해서만 지원됩니다.")
            else:
                tuning_epochs_to_use = args.tuning_epochs if args.tuning_epochs is not None else args.epochs
                run_hyperparameter_tuning(
                    data_yaml_path=custom_yaml_path,
                    n_trials=args.n_trials,
                    base_project_dir=args.project, # results 등 기본 프로젝트 경로
                    study_name=f"yolov8_{args.yolo_model_size_for_tuning}_tuning",
                    base_epochs=tuning_epochs_to_use,
                    model_size=args.yolo_model_size_for_tuning,
                    imgsz=args.imgsz,
                    batch_size=args.batch_size,
                    device=str(device), # device 전달 (문자열로 변환)
                    workers=args.workers # workers 전달
                )
            print("--- 하이퍼파라미터 튜닝 종료 ---")
            # 튜닝 후에는 일반적으로 프로그램 종료 (별도 학습/추론을 원하면 플래그 조절 필요)
            return # 튜닝 후 종료

        if args.train:
            resume_ckpt = find_resume_checkpoint(args.resume)
            print(f"[INFO][main] 실제로 train_model에 전달되는 resume 체크포인트: {resume_ckpt}")
            model = train_model(
                data_yaml_path=custom_yaml_path,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch_size=args.batch_size,
                project=args.project,
                device=str(device),
                workers=args.workers,
                model_size="s",
                resume=resume_ckpt
            )
        elif args.weights:
            print(f"1. YOLOv8 모델 로드: {args.weights}")
            model = YOLOModel(device=device) # device 전달
            model.load(args.weights)
    
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
                val_img_root=args.val_img_root,
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
        
        # 추론 전 테스트 이미지에 on-the-fly 전처리 적용
        # generate_prediction 함수 내부에서 이미지 로드 후 전처리 적용하도록 수정 필요
        # 여기서는 generate_prediction 함수에 preprocess=True와 같은 인자를 추가한다고 가정합니다.
        # 또는, generate_prediction 호출 전에 test_dir의 이미지들을 임시로 전처리하고 그 경로를 넘길 수도 있습니다.
        # 일단은 generate_prediction 내부 수정을 가정.
        out_label_dir = generate_prediction(model, args.test_dir, preprocess_on_the_fly=True) 
        
        # 5) 평가: 감지 정확도 계산 (정밀도, 재현율, F1 점수)
        print("5. 평가 수행 (IOU 임계값: 0.5)")
        metrics = evaluate_detection(f"{args.test_dir}/labels_yolo", out_label_dir, debug=True)
        
        print(
            f"종합 평가 결과 - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
            f" mAP@0.5: {metrics['mAP@0.5']:.4f}"
        )

if __name__ == "__main__":
    main()
