import optuna
import os
import yaml
import torch
from src.train import train_model # train_model 함수 임포트
from models.yolo_v8 import YOLOModel # YOLOModel 임포트 (클래스 이름 가져오기용)

def objective(
    trial: optuna.Trial,
    data_yaml_path: str,
    base_project_dir: str,
    base_epochs: int = 50, # 튜닝 시 기본 에포크 (Hyperband가 조절 가능)
    model_size: str = "s",
    imgsz: int = 640,
    batch_size: int = None, # None이면 YOLO가 자동 결정
    device: str = "auto",
    workers: int = 8 # workers 인자 추가
):
    """
    Optuna objective 함수.
    주어진 trial에 대해 모델을 학습하고 평가 점수를 반환합니다.
    """
    # 결과 저장 디렉토리 설정 (각 trial마다 다른 디렉토리 사용)
    trial_project_name = f"trial_{trial.number}"
    current_project_dir = os.path.join(base_project_dir, "optuna_trials") # 기본 results 하위에 optuna_trials 생성
    # trial별 name은 YOLO 내부에서 생성 (예: train, train2, ...)
    # YOLOModel의 train에서 project와 name을 인자로 받으므로, objective 레벨에서 name을 직접 설정하지 않아도 됨
    # 다만, 모든 trial 결과가 동일한 project/name 하위에 저장되는 것을 피하려면
    # YOLOModel.train에 전달하는 name을 trial별로 다르게 하는 것이 좋음. 
    # 여기서는 train_model 함수가 project 인자를 받으므로, trial마다 고유한 project 경로를 전달.
    # Ultralytics는 project/name 구조를 사용.
    # project = current_project_dir, name = trial_project_name 으로 전달하면 results/optuna_trials/trial_X/weights/best.pt 형태
    
    # 하이퍼파라미터 제안
    # 예시: 학습률, 가중치 감쇠, 증강 관련 파라미터 등
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)
    lrf = trial.suggest_float("lrf", 0.01, 1.0, log=True) # lr0 대비 최종 학습률 비율
    momentum = trial.suggest_float("momentum", 0.8, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    warmup_epochs = trial.suggest_float("warmup_epochs", 1.0, 5.0)
    warmup_momentum = trial.suggest_float("warmup_momentum", 0.5, 0.95)
    warmup_bias_lr = trial.suggest_float("warmup_bias_lr", 0.05, 0.2)
    
    # 손실 함수 가중치 (기본값 사용을 위해 주석 처리)
    # box = trial.suggest_float("box", 5.0, 10.0)
    # cls = trial.suggest_float("cls", 0.3, 0.7)
    # dfl = trial.suggest_float("dfl", 1.0, 2.0)

    # 데이터 증강 파라미터 (YOLO 내부 증강 사용 시)
    # augment=True는 train_model의 기본값이므로 여기서는 제어하지 않음.
    # 필요하다면 trial.suggest_categorical('augment', [True, False]) 등으로 추가 가능
    mosaic = trial.suggest_float("mosaic", 0.0, 1.0) # 모자이크 확률
    mixup = trial.suggest_float("mixup", 0.0, 0.5)   # 믹스업 확률
    degrees = trial.suggest_float("degrees", 0.0, 45.0) # 회전
    translate = trial.suggest_float("translate", 0.0, 0.3) # 이동
    scale = trial.suggest_float("scale", 0.1, 0.9) # 크기 조절 (слишком большие значения могут привести к ошибкам)
    shear = trial.suggest_float("shear", 0.0, 10.0) # 전단
    perspective = trial.suggest_float("perspective", 0.0, 0.001)
    flipud = trial.suggest_float("flipud", 0.0, 0.5) # 상하 반전
    fliplr = trial.suggest_float("fliplr", 0.0, 0.5) # 좌우 반전 (기본 0.5이므로 조정 가능)

    try:
        print(f"\n--- Optuna Trial {trial.number} 시작 ---")
        print(f"  하이퍼파라미터: {trial.params}")
        print(f"  Data YAML: {data_yaml_path}")
        print(f"  결과 저장 기본 경로: {current_project_dir}")
        print(f"  Trial별 프로젝트명: {trial_project_name}")
        
        # 모델 학습
        # project 경로를 trial별로 고유하게 전달
        # train_model의 project 인자는 전체 실행의 루트 디렉토리(예: 'results')를 받고,
        # YOLOModel.train의 project와 name 인자를 통해 세부 경로가 결정됨.
        # train_model 내부에서 YOLOModel을 초기화하고 train을 호출하므로, 
        # train_model에 project와 name을 모두 전달하거나, train_model이 이를 조합해야 함.
        # 여기서는 YOLOModel.train이 project=current_project_dir, name=trial_project_name 으로 호출되도록 수정되었다고 가정.
        # 또는 train_model에 trial_project_name과 같은 고유 식별자를 전달하여 내부적으로 사용.
        # train_model의 project 인자를 current_project_dir/trial_project_name 으로 설정.
        full_trial_project_path = os.path.join(current_project_dir, trial_project_name)

        # 학습 에포크는 Hyperband Pruner에 의해 동적으로 조절될 수 있음 (조기 중단)
        # trainer.py의 train_model은 epochs 인자를 그대로 사용하므로, Pruner는 콜백을 통해 제어
        training_results = train_model(
            data_yaml_path=data_yaml_path,
            epochs=base_epochs, # Pruner가 이 epoch 내에서 조기 중단 결정
            model_size=model_size,
            imgsz=imgsz,
            batch_size=batch_size,
            device=device,
            workers=workers, # workers 인자 전달
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            # box=box, # 기본값 사용
            # cls=cls, # 기본값 사용
            # dfl=dfl, # 기본값 사용
            augment=True, # 기본적으로 Ultralytics 내부 증강 사용
            mosaic=mosaic,
            mixup=mixup,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            project=full_trial_project_path, # 각 trial의 결과를 저장할 고유 경로
            # optuna_trial=trial # Optuna trial 객체 전달 (프루닝용) -> 주석 처리
        )

        # 평가 지표 반환 (예: mAP@0.5)
        # training_results는 Ultralytics의 Results 객체
        # results.fitness는 val mAP50-B를 의미할 수 있음 (YOLOv8 기본)
        # 또는 results.metrics 딕셔너리에서 직접 접근
        if training_results is None:
            print(f"경고: Trial {trial.number}의 training_results가 None입니다. 학습 중 오류 발생 가능성. 반환값 0.0")
            return 0.0
            
        map50 = training_results.metrics.get('metrics/mAP50-B') # 가장 일반적인 mAP@0.5 키
        if map50 is None:
            map50 = training_results.fitness # 백업으로 fitness 사용
        
        if map50 is None:
            print(f"경고: Trial {trial.number}에서 mAP50-B 또는 fitness를 찾을 수 없습니다. 반환값 0.0")
            return 0.0 # 실패 또는 메트릭 부재 시 낮은 값 반환
            
        print(f"--- Optuna Trial {trial.number} 종료 --- mAP@0.5: {map50:.4f}")
        return float(map50) # Optuna는 float 반환값을 기대

    except optuna.TrialPruned as e:
        print(f"Optuna Trial {trial.number} Pruned: {e}")
        raise # Optuna가 처리하도록 예외 다시 발생
    except Exception as e:
        print(f"Optuna Trial {trial.number} 실패: {e}")
        import traceback
        traceback.print_exc()
        return 0.0  # 예외 발생 시 낮은 점수 반환 (실패로 간주)

def run_hyperparameter_tuning(
    data_yaml_path: str, 
    n_trials: int = 100, 
    base_project_dir: str = "results", # 기본 결과 저장 경로
    study_name: str = "yolov8_tuning",
    storage_name: str = "sqlite:///yolov8_tuning.db", # SQLite DB에 저장
    base_epochs: int = 50, # 각 trial의 최대 에포크 (Pruner가 조절)
    model_size: str = "s",
    imgsz: int = 640,
    batch_size: int = None,
    device: str = "auto",
    workers: int = 8 # workers 인자 추가
):
    """
    Optuna를 사용하여 하이퍼파라미터 튜닝을 실행합니다.
    """
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"데이터 YAML 파일을 찾을 수 없습니다: {data_yaml_path}")

    # study_name과 storage_name을 조합하여 고유한 DB 경로 생성
    # 예: "sqlite:///results/optuna_trials/yolov8_tuning.db"
    optuna_storage_dir = os.path.join(base_project_dir, "optuna_trials")
    os.makedirs(optuna_storage_dir, exist_ok=True)
    actual_storage_name = f"sqlite:///{os.path.join(optuna_storage_dir, f'{study_name}.db')}"

    print(f"Optuna 스터디 시작: {study_name}")
    print(f"  저장소: {actual_storage_name}")
    print(f"  Trial 수: {n_trials}")
    print(f"  Pruner: HyperbandPruner")

    # HyperbandPruner 설정
    # min_resource: 최소 리소스 (예: 최소 에포크). 기본값 1.
    # max_resource: 최대 리소스 (예: 'auto' 또는 base_epochs).
    # reduction_factor: 각 라운드에서 제거할 trial 비율의 역수. 기본값 3.
    # pruner = optuna.pruners.HyperbandPruner(
    #     min_resource=max(1, base_epochs // 10), # 최소 1 에포크 또는 전체의 10%
    #     max_resource=base_epochs, # objective의 epochs와 일치시킴
    #     reduction_factor=3
    # )
    pruner = None # Pruner 비활성화
    
    study = optuna.create_study(
        study_name=study_name,
        storage=actual_storage_name, # study 진행 상황 저장
        direction="maximize", # mAP@0.5 최대화
        pruner=pruner,
        load_if_exists=True # 동일 이름의 스터디가 있으면 이어서 진행
    )

    # 큐에 이미 완료된 trial이 있다면, 남은 trial만 실행
    n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = n_trials - n_completed_trials

    if remaining_trials <= 0 and n_trials > 0:
        print(f"스터디 '{study_name}'은 이미 {n_completed_trials}개의 Trial을 완료했습니다. 추가 실행이 없습니다.")
    elif n_trials > 0:
        if n_completed_trials > 0:
            print(f"이전에 {n_completed_trials}개의 Trial을 완료했습니다. {remaining_trials}개의 Trial을 추가로 실행합니다.")
        study.optimize(
            lambda trial: objective(
                trial,
                data_yaml_path=data_yaml_path,
                base_project_dir=base_project_dir,
                base_epochs=base_epochs,
                model_size=model_size,
                imgsz=imgsz,
                batch_size=batch_size,
                device=device,
                workers=workers
            ),
            n_trials=remaining_trials, # 남은 trial 수만큼 실행
            # n_jobs=1, # 병렬 실행 (GPU 사용 시 주의, 보통 1로 설정)
            gc_after_trial=True # 각 trial 후 가비지 컬렉션 (메모리 누수 방지)
        )

    print(f"\nOptuna 스터디 '{study_name}' 완료.")
    print(f"  총 Trial 수: {len(study.trials)}")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Complete trials: {len(complete_trials)}")

    if complete_trials: # 성공적으로 완료된 trial이 있는 경우에만 최적 결과 출력
        print(f"\n최적화 결과:")
        print(f"  최고 점수 (mAP@0.5): {study.best_value:.4f}")
        print(f"  최적 하이퍼파라미터:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        print(f"  최적 Trial 번호: {study.best_trial.number}")
        # 최적 가중치 경로 등 추가 정보가 필요하면 objective 함수에서 저장하고 여기서 로드할 수 있음
        best_trial_project_path = os.path.join(base_project_dir, "optuna_trials", f"trial_{study.best_trial.number}")
        print(f"  최적 Trial 결과 저장 경로: {best_trial_project_path}")
    else:
        print("완료된 Trial이 없어 최적 결과를 표시할 수 없습니다.")

    return study

# 이 스크립트가 직접 실행될 때 (테스트용)
if __name__ == '__main__':
    # 실제 사용 시에는 main.py에서 이 함수를 호출합니다.
    # 테스트를 위해서는 실제 data.yaml 경로와 프로젝트 디렉토리 설정이 필요합니다.
    print("Hyperparameter tuning 스크립트 직접 실행 (테스트 모드)")
    
    # 임시 data.yaml 생성 (테스트용)
    # 실제로는 main.py에서 생성된 custom_data.yaml 경로를 전달받아야 함
    mock_data_yaml_content = {
        "train": "../data/train/images_processed", # 실제 경로로 수정 필요
        "val": "../data/val/images_processed",     # 실제 경로로 수정 필요
        "names": {0: "class1", 1: "class2"}, # 실제 클래스 정보로 수정 필요
        "nc": 2
    }
    mock_yaml_path = "temp_mock_data.yaml"
    with open(mock_yaml_path, "w") as f:
        yaml.dump(mock_data_yaml_content, f)
    print(f"임시 Mock data.yaml 생성: {mock_yaml_path}")

    # 기본 설정으로 튜닝 실행 (n_trials 줄여서 테스트)
    # 이 테스트는 CUDA가 사용 가능하고, Ultralytics 및 PyTorch가 올바르게 설치된 환경에서 실행해야 함.
    try:
        run_hyperparameter_tuning(
            data_yaml_path=mock_yaml_path,
            n_trials=3, # 테스트를 위해 trial 수 줄임
            base_project_dir="results_tuning_test",
            study_name="yolov8_test_tuning",
            storage_name="sqlite:///yolov8_test_tuning.db", # 테스트용 DB
            base_epochs=10, # 테스트를 위해 에포크 수 줄임
            model_size="n", # 가장 작은 모델로 테스트
            imgsz=320, # 이미지 크기 줄여서 테스트
            batch_size=None,
            device="auto",
            workers=0 # 테스트 시 워커 수 0으로 설정 (메모리 문제 방지)
        )
    except Exception as e:
        print(f"테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(mock_yaml_path):
            os.remove(mock_yaml_path)
            print(f"임시 Mock data.yaml 삭제: {mock_yaml_path}")

    print("Hyperparameter tuning 스크립트 직접 실행 완료 (테스트 모드)") 