import os
import cv2
import random
import shutil
import time
from collections import defaultdict

from ultralytics import YOLO
# from ultralytics.engine.callbacks import BaseCallback # 이전 경로 주석 처리
# from ultralytics.utils.callbacks.base import BaseCallback # BaseCallback 임포트 제거
import optuna


# Optuna Pruning 콜백 생성을 위한 헬퍼 함수
# def create_optuna_pruning_callback_fn(trial: optuna.Trial, metric_key: str = "metrics/mAP50-B"):
#     """
#     Optuna Pruning을 위한 콜백 함수를 생성하여 반환합니다.
#     반환된 함수는 Ultralytics의 'on_fit_epoch_end' 이벤트에 등록될 수 있습니다.
#
#     매개변수:
#         trial (optuna.Trial): 현재 Optuna trial 객체.
#         metric_key (str): Pruning에 사용할 metric의 키 (예: "metrics/mAP50-B").
#
#     반환값:
#         function: Ultralytics Trainer를 인자로 받는 콜백 함수.
#     """"
#
#     def on_epoch_end_pruning_callback(trainer):
#         """"
#         실제 Pruning 로직을 수행하는 콜백 함수.
#         Trainer 객체에서 메트릭을 가져와 Optuna trial에 보고하고, 필요한 경우 Pruning을 수행합니다.
#         """"
#         epoch_num = trainer.epoch + 1  # Optuna는 1-based step을 선호
#         current_value = trainer.metrics.get(metric_key)
#
#         if current_value is None:
#             # 메트릭 키가 없을 경우, 학습 초기이거나 'fitness'를 사용해야 할 수 있음
#             current_value = trainer.fitness  # trainer.fitness는 일반적으로 주요 메트릭을 나타냄
#             if current_value is None:
#                 print(f"경고: Optuna 콜백에서 메트릭 '{metric_key}' 또는 'fitness'를 찾을 수 없습니다. Epoch: {epoch_num}")
#                 # Pruning 결정을 내릴 수 없으므로 일단 진행
#                 return
#
#         trial.report(current_value, epoch_num)
#
#         if trial.should_prune():
#             message = f"Trial {trial.number} pruned at epoch {epoch_num} with {metric_key}: {current_value:.4f}."
#             # Ultralytics 학습 루프는 optuna.TrialPruned 예외를 직접 처리하지 않을 수 있으므로,
#             # 여기서는 학습을 중단시키는 다른 방법을 고려하거나, 단순히 메시지를 남기고
#             # Optuna가 다음 trial로 넘어가도록 할 수 있습니다.
#             # 일단은 Optuna의 표준 예외를 발생시킵니다.
#             # Ultralytics의 Trainer가 이 예외를 어떻게 처리할지는 확인 필요.
#             # 만약 처리하지 못한다면, trainer.stop() 와 같은 플래그를 설정해야 할 수도 있습니다.
#             print(f"Optuna Pruning: {message}") # 로그 추가
#             raise optuna.TrialPruned(message)
#
#     return on_epoch_end_pruning_callback


class YOLOModel:       
    """
    YOLOv8(You Only Look Once) 모델을 위한 래퍼 클래스입니다.

    이 클래스는 Ultralytics 라이브러리의 YOLO 구현체를 활용하여 다음 기능을 제공합니다:
    1. 사전 훈련된 YOLO 모델 로드 또는 커스텀 데이터셋으로 학습
    2. 이미지에서 객체 감지 수행
    3. 학습된 모델 저장 및 로드

    YOLO는 단일 스테이지 객체 감지 알고리즘으로, 빠른 추론 속도와 높은 정확도를 제공합니다.
    """

    DEFAULT_CLASS_NAMES = [
        "경차/세단", "SUV/승합차", "트럭", "버스(소형, 대형)",
        "통학버스(소형,대형)", "경찰차", "구급차", "소방차", "견인차",
        "기타 특장차", "성인", "어린이", "오토바이", "자전거 / 기타 전동 이동체",
        "라바콘", "삼각대", "기타"
    ]

    def __init__(self, model_name="yolov8n.pt", device="auto", class_names=None):
        """
        YOLOv8 모델을 초기화합니다.

        매개변수:
            model_name (str): 사용할 모델의 이름 또는 경로
                - "yolov8n.pt": 나노 모델 (가장 작고 빠름, 기본값)
                - "yolov8s.pt": 소형 모델 (속도와 정확도 균형)
                - "yolov8m.pt": 중형 모델 (중간 성능)
                - "yolov8l.pt": 대형 모델 (높은 정확도)
                - "yolov8x.pt": 초대형 모델 (최고 정확도)
                - 커스텀 학습된 모델의 경로 (예: "runs/detect/exp/weights/best.pt")
            device (str): 모델 실행 장치
                - "auto": 자동 선택 (CUDA 사용 가능시 GPU, 아니면 CPU)
                - "cpu": CPU만 사용
                - "cuda": GPU 사용 (CUDA 지원 필요)
                - "0" 또는 "1": 특정 GPU 장치 지정
            class_names (list): 사용자 정의 클래스 이름 목록

        참고:
            - 초기화 시점에는 실제 모델이 메모리에 로드되지 않으며, 첫 predict() 호출 시 로드됩니다.
            - 큰 모델일수록 더 정확하지만 속도가 느리고 메모리 사용량이 증가합니다.
        """
        self.model_name = model_name  # 모델 가중치 파일 이름 또는 경로
        self.device = device  # 실행 장치 (CPU/GPU)
        self.best_weight = None  # 학습 후 생성된 최적 가중치 경로
        self.model = None  # 실제 YOLO 모델 객체 (지연 로딩)
        self.class_names = class_names if class_names is not None else self.DEFAULT_CLASS_NAMES
        # YOLO 객체는 train/predict/save/load 등에서 동적으로 생성

    def get_class_names(self):
        """ 현재 모델 인스턴스에 설정된 클래스 이름 목록을 반환합니다. """
        return self.class_names

    def _create_data_yaml(self, train_dir, val_dir, class_names):
        """
        YOLO 학습에 필요한 data.yaml 파일을 생성합니다.

        매개변수:
            train_dir (str): 학습 데이터 디렉토리 경로
            val_dir (str): 검증 데이터 디렉토리 경로
            class_names (list): 클래스 이름 목록

        반환값:
            str: data.yaml 파일 경로
        """
        import yaml

        # 상위 디렉토리 결정
        parent_dir = os.path.dirname(train_dir) if os.path.dirname(train_dir) else "."
        yaml_path = os.path.join(parent_dir, "data.yaml")

        # 절대 경로로 변환
        train_dir_abs = os.path.abspath(train_dir)
        val_dir_abs = os.path.abspath(val_dir) if val_dir else train_dir_abs

        # 클래스 이름이 없으면 기본값 사용
        if not class_names:
            class_names = ["object"]

        # YAML 데이터 구성
        data = {
            "train": train_dir_abs,
            "val": val_dir_abs,
            "names": {i: name for i, name in enumerate(class_names)},
            "nc": len(class_names),
        }

        # YAML 파일 작성
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"[YOLOModel] data.yaml 파일 생성됨: {yaml_path}")
        print(f"[YOLOModel] YAML 내용:\n{data}")

        return yaml_path

    def train(self, data_yaml_path, epochs, batch_size, imgsz, device, resume=None, optuna_trial=None, **kwargs):
        """
        Trains the YOLOv8 model.

        Args:
            data_yaml_path (str): Path to the data YAML file.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            imgsz (int): Image size for training.
            device (str): Device to use for training (e.g., 'cpu', '0', '0,1').
            resume (bool, optional): Whether to resume training from a previous checkpoint. Defaults to None.
            optuna_trial (optuna.trial.Trial, optional): Optuna trial for hyperparameter tuning. Defaults to None.
            **kwargs: Additional hyperparameters for training.
        """
        # 장치 설정: 'auto' 또는 CUDA 사용 가능 여부에 따라 CPU 또는 GPU 선택
        # if device == 'auto':
        #     resolved_device = '0' if torch.cuda.is_available() else 'cpu'
        # elif device != 'cpu' and not torch.cuda.is_available():
        #     print(f"Warning: CUDA device '{device}' requested but not available. Using 'cpu' instead.")
        #     resolved_device = 'cpu'
        # else:
        #     resolved_device = device
        
        resolved_device = device # device 파라미터를 직접 사용하도록 변경
        print(f"Training on device: {resolved_device}")

        # resume 체크포인트가 있으면, epochs 인자를 강제로 override
        if resume:
            import torch
            try:
                ckpt = torch.load(resume, map_location='cpu', weights_only=False)
                if 'train_args' in ckpt and 'epochs' in ckpt['train_args']:
                    print(f"[DEBUG] resume 체크포인트에서 epochs={ckpt['train_args']['epochs']} → 명령행 인자 epochs={epochs}로 강제 세팅")
                else:
                    print(f"[DEBUG] resume 체크포인트에서 epochs 정보 없음, 명령행 인자 epochs={epochs} 사용")
            except Exception as e:
                print(f"[경고] resume 체크포인트에서 epochs 정보 읽기 실패: {e}")
            # 어쨌든 명령행 인자 epochs를 사용하도록 강제

        # 기본 하이퍼파라미터 설정
        lr0 = kwargs.get('lr0', 0.01)
        lrf = kwargs.get('lrf', 0.01)
        momentum = kwargs.get('momentum', 0.937)
        weight_decay = kwargs.get('weight_decay', 0.0005)
        warmup_epochs = kwargs.get('warmup_epochs', 3.0)
        warmup_momentum = kwargs.get('warmup_momentum', 0.8)
        warmup_bias_lr = kwargs.get('warmup_bias_lr', 0.1)
        box = kwargs.get('box', 7.5)
        cls = kwargs.get('cls', 0.5)
        dfl = kwargs.get('dfl', 1.5)
        augment = kwargs.get('augment', True)
        mosaic = kwargs.get('mosaic', 1.0)
        mixup = kwargs.get('mixup', 0.0)
        degrees = kwargs.get('degrees', 0.0)
        translate = kwargs.get('translate', 0.1)
        scale = kwargs.get('scale', 0.5)
        shear = kwargs.get('shear', 0.0)
        perspective = kwargs.get('perspective', 0.0)
        flipud = kwargs.get('flipud', 0.0)
        fliplr = kwargs.get('fliplr', 0.5)

        # 임시로 Pruning 콜백 비활성화
        # if optuna_trial:
        #     print("Optuna trial detected, adding pruning callback.")
        #     pruning_callback_fn = create_optuna_pruning_callback_fn(optuna_trial)
        #     self.model.add_callback("on_fit_epoch_end", pruning_callback_fn)
        # else:
        #     print("No Optuna trial detected or pruning disabled.")

        try:
            print(f"Starting training with a {epochs} epochs, batch size {batch_size}, image size {imgsz}...")
            print(f"Hyperparameters: {kwargs}")
            # YOLO 객체 동적 생성 (resume이 있으면 해당 pt로, 없으면 model_name으로)
            from ultralytics import YOLO
            if resume and os.path.exists(resume):
                try:
                    self.model = YOLO(resume)
                    print(f"[YOLOModel.train] resume 체크포인트로 모델 로드: {resume}")
                except Exception as e:
                    raise RuntimeError(f"[YOLOModel.train] resume 체크포인트 모델 로드 실패: {resume}, 오류: {e}")
            else:
                try:
                    self.model = YOLO(self.model_name)
                    print(f"[YOLOModel.train] 기본 모델로 로드: {self.model_name}")
                except Exception as e:
                    raise RuntimeError(f"[YOLOModel.train] 기본 모델 로드 실패: {self.model_name}, 오류: {e}")
            results = self.model.train(
                data=data_yaml_path,
                epochs=epochs,  # 무조건 인자로 받은 epochs 사용!
                batch=batch_size,
                imgsz=imgsz,
                device=resolved_device, # 수정된 장치 사용
                resume=resume,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                warmup_momentum=warmup_momentum,
                warmup_bias_lr=warmup_bias_lr,
                box=box,
                cls=cls,
                dfl=dfl,
                augment=augment,
                mosaic=mosaic,
                mixup=mixup,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                flipud=flipud,
                fliplr=fliplr
            )
            print("Training completed.")
            # 학습 완료 후 콜백 제거 (필요한 경우)
            # if optuna_trial:
            #     self.model.clear_callbacks()
            return results
        except Exception as e:
            print(f"Error during YOLO model training: {e}")
            # 에러 발생 시 콜백 제거 (필요한 경우)
            # if optuna_trial:
            #     self.model.clear_callbacks()
            
            # Optuna 사용 시 예외를 다시 발생시켜 Optuna가 처리하도록 함
            if optuna_trial:
                # 특정 유형의 오류는 Optuna에 의해 다르게 처리될 수 있음
                # 예를 들어, 사용자가 중단한 경우 Pruned 예외를 발생시킬 수 있음
                if isinstance(e, KeyboardInterrupt):
                    raise optuna.exceptions.TrialPruned("Training interrupted by user.")
                # 여기서 None 대신 에러를 직접 발생시키거나, Optuna가 실패로 인지할 값을 반환할 수 있습니다.
                # 일반적으로는 에러를 그대로 전파하거나, 특수한 경우 optuna.exceptions.TrialPruned를 발생시킵니다.
                # raise e # 에러를 그대로 전파하여 Optuna가 실패로 기록하도록 함
                print(f"Error for Optuna: {e}") # 로깅은 하되, Optuna는 mAP 0.0으로 실패처리하도록 None 반환 유지
            return None

    def predict(self, img_bgr, conf_thresh=0.25):
        """
        이미지에서 객체를 감지합니다.

        매개변수:
            img_bgr (ndarray): OpenCV BGR 형식의 입력 이미지
            conf_thresh (float): 객체 감지 신뢰도 임계값 (0.0 ~ 1.0)

        반환값:
            list: 감지된 객체의 바운딩 박스 목록
                - 각 항목은 [x1, y1, x2, y2, score, class_id] 형식
                - class_id: 객체 클래스 ID (0-16)
        """
        if self.model is None:
            if self.best_weight and os.path.exists(self.best_weight):
                print(f"[YOLOModel.predict] 학습된 가중치 로드: {self.best_weight}")
                self.model = YOLO(self.best_weight)
            elif isinstance(self.model_name, str) and os.path.exists(self.model_name):
                print(f"[YOLOModel.predict] 지정된 모델 경로에서 로드: {self.model_name}")
                self.model = YOLO(self.model_name)
            else:
                # 기본 모델 이름(e.g., "yolov8n.pt")으로 시도
                try:
                    print(f"[YOLOModel.predict] 사전 훈련된 모델 로드 시도: {self.model_name if isinstance(self.model_name, str) else 'yolov8n.pt'}")
                    self.model = YOLO(self.model_name if isinstance(self.model_name, str) else "yolov8n.pt")
                except Exception as e:
                    raise RuntimeError(f"[YOLOModel.predict] 모델 로드 실패 ({self.model_name}): {e}")
        
        results = self.model.predict(img_bgr, conf=conf_thresh, device=self.device)
        result = results[0] 

        boxes = []
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes.data:
                box_data = box.cpu().numpy()
                x1, y1, x2, y2 = map(float, box_data[:4])
                conf = float(box_data[4])
                class_id = int(box_data[5]) if len(box_data) > 5 else 0
                boxes.append([x1, y1, x2, y2, conf, class_id])
        return boxes

    def save(self, path):
        """
        현재 로드된 YOLO 모델을 파일로 저장합니다.

        매개변수:
            path (str): 모델을 저장할 파일 경로
                - 일반적으로 .pt 확장자 사용
                - 경로에 존재하지 않는 디렉토리가 있으면 자동 생성

        예외:
            RuntimeError: 모델이 로드되지 않은 상태에서 호출 시 발생

        참고:
            - 모델은 가중치, 구조, 하이퍼파라미터를 포함하여 저장됨
            - 저장된 모델은 load() 메서드로 다시 로드 가능
        """
        if self.model is None:
            raise RuntimeError(
                "[YOLOModel.save] 저장할 모델이 없습니다. 먼저 학습하거나 모델을 로드하세요."
            )

        # 파일 확장자 확인 및 추가
        if not path.endswith(".pt"):
            path += ".pt"

        # 상위 디렉토리 생성 (없는 경우)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # 모델 저장
        self.model.save(path)
        print(f"[YOLOModel.save] 모델 저장 완료: {path}")

    def load(self, path):
        """
        저장된 YOLO 모델을 파일에서 로드합니다.

        매개변수:
            path (str): 로드할 모델 파일 경로
                - .pt 확장자의 파일이어야 함
                - 사전 훈련된 모델명(예: "yolov8n.pt") 또는 경로 가능

        예외:
            FileNotFoundError: 지정된 경로에 모델 파일이 없을 경우 발생

        참고:
            - 이전에 로드된 모델이 있다면 메모리에서 해제하고 새 모델 로드
            - 학습/추론에 사용할 모델을 명시적으로 지정할 때 유용
        """
        if not os.path.exists(path):
            # path가 'yolov8n.pt'와 같은 공식 이름일 수도 있음.
            try:
                self.model = YOLO(path)
                self.model_name = path
                self.best_weight = path if Path(path).is_file() else None # 로드된 경로가 파일이면 best_weight로 간주
                print(f"[YOLOModel.load] 모델 로드 완료 (Ultralytics): {path}")
                return
            except Exception as e_load:
                raise FileNotFoundError(f"[YOLOModel.load] 모델 파일을 찾을 수 없거나 로드 실패: {path}. 오류: {e_load}")

        # 경로가 존재하면 YOLO로 로드 시도
        try:
            self.model = YOLO(path)
            self.model_name = path
            self.best_weight = path # 로드 성공 시 이 경로를 best_weight로 간주
            print(f"[YOLOModel.load] 모델 로드 완료: {path}")
        except Exception as e:
            raise RuntimeError(f"[YOLOModel.load] 모델 로드 중 오류 ({path}): {e}")


def create_subset(source_dir, target_dir, sampling_ratio=0.3, min_samples=10):
    """주요 클래스는 30% 샘플링, 소수 클래스는 전부 유지하는 전략"""
    os.makedirs(target_dir, exist_ok=True)
    label_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
    
    print(f"\n[데이터 축소] 원본 파일 수: {len(label_files)}개")
    print(f"[데이터 축소] 소스 경로: {source_dir}")
    print(f"[데이터 축소] 타겟 경로: {target_dir}")
    print(f"[데이터 축소] 샘플링 비율: {sampling_ratio*100:.1f}% (소수 클래스 100% 유지)")
    print("-" * 60)
    
    # 클래스별 파일 분류
    print("[데이터 축소] 파일 분석 중...")
    class_files = {}
    for i, file in enumerate(label_files):
        if i % 1000 == 0 and i > 0:
            print(f"  - {i}/{len(label_files)} 파일 처리 중... ({i/len(label_files)*100:.1f}%)")
            
        try:
            with open(os.path.join(source_dir, file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id not in class_files:
                            class_files[class_id] = []
                        class_files[class_id].append(file)
                        break  # 첫 객체의 클래스만 확인
        except Exception as e:
            print(f"  - 경고: 파일 {file} 처리 중 오류: {str(e)}")
    
    # 클래스별 샘플링
    selected_files = set()
    print("\n[데이터 축소] 클래스별 샘플링:")
    print(f"{'클래스ID':<10}{'원본 수':<10}{'샘플 수':<10}{'비율':<10}")
    print("-" * 40)
    
    total_orig = 0
    total_sampled = 0
    
    for class_id, files in sorted(class_files.items()):
        total_orig += len(files)
        # 소수 클래스(경찰차 등)는 전부 포함
        if len(files) < 150:  # 소수 클래스 기준
            sample_size = len(files)
            rare_class = True
        else:
            sample_size = max(min_samples, int(len(files) * sampling_ratio))
            rare_class = False
        
        if len(files) <= sample_size:
            samples = files
        else:
            samples = random.sample(files, sample_size)
        
        selected_files.update(samples)
        total_sampled += len(samples)
        
        ratio = len(samples) / len(files) * 100 if len(files) > 0 else 0
        status = "전체유지" if rare_class else f"{ratio:.1f}%"
        print(f"{class_id:<10}{len(files):<10}{len(samples):<10}{status:<10}")
    
    # 결과 요약
    print("-" * 40)
    print(f"총계: {total_orig} -> {total_sampled} ({total_sampled/total_orig*100:.1f}%)")
    
    # 파일 복사
    print(f"\n[데이터 축소] {len(selected_files)}개 파일 복사 중...")
    start_time = time.time()
    
    for i, file in enumerate(selected_files):
        if (i+1) % 100 == 0 or (i+1) == len(selected_files):
            elapsed = time.time() - start_time
            files_per_sec = (i+1) / elapsed if elapsed > 0 else 0
            eta = (len(selected_files) - (i+1)) / files_per_sec if files_per_sec > 0 else 0
            progress = (i+1) / len(selected_files) * 100
            
            print(f"  - 진행: {progress:.1f}% ({i+1}/{len(selected_files)}) - "
                  f"속도: {files_per_sec:.1f} 파일/초, "
                  f"남은 시간: {eta:.1f}초")
        
        shutil.copy2(os.path.join(source_dir, file), target_dir)
    
    total_time = time.time() - start_time
    print(f"\n[데이터 축소] 완료: {len(selected_files)}개 파일 복사됨 (소요시간: {total_time:.2f}초)")
    return len(selected_files)
