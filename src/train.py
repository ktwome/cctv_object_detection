from models.yolo_v8 import YOLOModel # , create_optuna_pruning_callback_fn # 주석 처리
import optuna

# 클래스 이름을 모듈 레벨 또는 YOLOModel에서 직접 가져올 수 있도록 변경 고려
# 현재는 train_model 함수 내에 정의되어 있어 main.py에서 직접 사용하기 어려움
# 여기서는 YOLOModel.DEFAULT_CLASS_NAMES를 사용한다고 가정하고, YOLOModel 수정 시 이 부분을 반영

def train_model(
    data_yaml_path, # train_dir, val_dir 대신 data.yaml 경로를 받음
    epochs=100, 
    model_size="s",
    imgsz=640,
    batch_size=None,
    device="auto",
    project="results",
    workers=8, # workers 인자 추가
    resume=None,
    # Optuna 튜닝을 위한 하이퍼파라미터 전달
    lr0=0.01, 
    lrf=0.01, 
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # 데이터 증강 관련 하이퍼파라미터
    augment=True,
    mosaic=1.0,
    mixup=0.0,
    degrees=0.0, 
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    # Optuna trial 객체 (튜닝 시에만 전달됨)
    optuna_trial: optuna.Trial = None 
):
    """
    YOLOv8 모델을 학습합니다.

    매개변수:
        data_yaml_path (str): 학습 설정이 포함된 data.yaml 파일 경로
        epochs (int): 학습 에포크 수
        model_size (str): 모델 크기 ('n', 's', 'm', 'l', 'x')
        imgsz (int): 입력 이미지 크기
        batch_size (int): 배치 크기 (None이면 자동 설정)
        device (str): 학습에 사용할 장치 (예: "cpu", "cuda", "0")
        project (str): 결과 저장 디렉토리 경로
        workers (int): 데이터 로딩에 사용할 워커 수
        resume (str, optional): 이전 학습 결과를 로드할 경로
        lr0 (float): 초기 학습률
        lrf (float): 최종 학습률 비율
        momentum (float): 옵티마이저의 momentum 값
        weight_decay (float): 가중치 감소 값
        warmup_epochs (float): 학습률 증가 에포크 수
        warmup_momentum (float): 학습률 증가 시작 시의 momentum 값
        warmup_bias_lr (float): 학습률 증가 시작 시의 bias learning rate
        box (float): 박스 손실 가중치
        cls (float): 클래스 손실 가중치
        dfl (float): 분포 초점 손실 가중치
        augment (bool): 데이터 증강 활성화 여부 (Ultralytics 내부 증강 사용)
        mosaic (float): 모자이크 증강 비율 (0.0-1.0)
        mixup (float): 믹스업 증강 비율 (0.0-1.0)
        optuna_trial (optuna.Trial, optional): Optuna trial 객체. 제공되면 프루닝 콜백이 활성화됩니다.

    반환값:
        학습 결과 객체 (Ultralytics Results 객체)
    """

    model_name = f"yolov8{model_size}.pt"
    
    # YOLOModel 초기화 시 클래스 이름 전달 (main.py에서 data.yaml 생성 시 사용한 것과 동일해야 함)
    # data.yaml에 이미 클래스 정보가 있으므로, YOLOModel 생성자에서 class_names를 필수로 받지 않아도 됨
    # 또는, data.yaml 경로를 전달하여 YOLOModel 내부에서 클래스 정보를 로드하도록 할 수도 있음
    # 여기서는 YOLOModel이 기본 클래스 이름을 가지거나, data.yaml을 통해 설정된다고 가정
    yolom = YOLOModel(model_name=model_name, device=device) # class_names 인자 제거 또는 YOLOModel 내부 로직에 따라 조정, device 전달
    
    # Optuna 콜백 설정 (주석 처리)
    # yolo_callbacks = None # callbacks를 None으로 초기화
    # if optuna_trial:
    #     pruning_callback_fn = create_optuna_pruning_callback_fn(optuna_trial)
    #     # Ultralytics YOLO는 콜백을 {'event_name': [callback_fn1, ...]} 형태의 딕셔너리로 받음
    #     yolo_callbacks = {'on_fit_epoch_end': [pruning_callback_fn]}
    #     # 튜닝 시에는 각 trial마다 고유한 디렉토리에 저장되도록 project/name 수정
    #     # 예: results/trial_0, results/trial_1 ...
    #     # 또는 objective 함수에서 trial.number를 사용하여 name을 생성
    #     # 여기서는 일단 project만 유지하고, 필요시 objective에서 name을 변경하도록 함.

    results = yolom.train(
        data_yaml_path=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        device=device,
        workers=workers, # workers 인자 전달
        resume=resume,
        # 학습률 및 옵티마이저
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        # 손실 함수 가중치
        box=box,
        cls=cls,
        dfl=dfl,
        # 데이터 증강
        augment=augment, # Ultralytics 내부의 augment 파라미터
        mosaic=mosaic,
        mixup=mixup,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        project=project,
        callbacks=None # yolo_callbacks # 콜백 전달 안 함
    )

    return results # YOLOModel.train에서 반환된 results 객체를 그대로 반환
