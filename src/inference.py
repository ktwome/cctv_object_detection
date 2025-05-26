import os
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from src.data_preprocessing import apply_custom_preprocessing


def preprocess_image(img, use_gray=False, use_clahe=True):
    """
    객체 감지 추론을 위한 이미지 전처리 함수

    매개변수:
        img (numpy.ndarray): 입력 이미지 (BGR 형식)
        use_gray (bool): 그레이스케일 변환 적용 여부
        use_clahe (bool): CLAHE 대비 향상 적용 여부

    반환값:
        numpy.ndarray: 전처리된 이미지 (BGR 형식)
    """
    if img is None:
        raise ValueError("[preprocess_image] 이미지가 비어 있습니다")

    # 이미지 복사본 생성 (원본 변경 방지)
    processed_img = img.copy()

    # 그레이스케일 변환 (선택 사항)
    if use_gray:
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 3채널 형식 유지

    # CLAHE 적용 (선택 사항)
    if use_clahe:
        # YCrCb 색상 공간으로 변환 (Y: 밝기 채널)
        ycrcb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        # 밝기 채널에만 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        # 채널 합치기
        merged = cv2.merge((y, cr, cb))
        processed_img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    return processed_img


def inference_yolo(yolo_model, test_dir, out_label_dir=None, apply_preprocessing_on_the_fly=False):
    """
    YOLOv8 모델을 사용하여 테스트 이미지에서 객체 감지 추론을 수행합니다.

    클래스 ID:
    0: 경차/세단       1: SUV/승합차        2: 트럭
    3: 버스(소형, 대형) 4: 통학버스(소형,대형) 5: 경찰차
    6: 구급차         7: 소방차            8: 견인차
    9: 기타 특장차     10: 성인            11: 어린이
    12: 오토바이       13: 자전거/기타 전동 이동체  14: 라바콘
    15: 삼각대        16: 기타

    매개변수:
        yolo_model: YOLOv8 모델 인스턴스
        test_dir: 테스트 이미지 디렉토리 경로
        out_label_dir: 출력 라벨 디렉토리 경로 (기본값: test_dir/labels_pred)
        apply_preprocessing_on_the_fly (bool): 추론 시점에 커스텀 전처리 적용 여부

    반환값:
        출력 라벨 디렉토리 경로
    """
    if out_label_dir is None:
        out_label_dir = os.path.join(test_dir, "labels_pred")

    os.makedirs(out_label_dir, exist_ok=True)
    img_dir = os.path.join(test_dir, "images")
    files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg"))]
    )

    print(f"[inference_yolo] 테스트 이미지: {len(files)}개")

    for f in files:
        # 이미지 경로 및 출력 파일 경로 생성
        imgp = os.path.join(img_dir, f)
        base = os.path.splitext(f)[0]
        out_txt = os.path.join(out_label_dir, base + ".txt")

        # 이미지 로드
        img = cv2.imread(imgp)
        if img is None:
            print(f"[경고] 이미지를 읽을 수 없습니다: {imgp}")
            continue

        # On-the-fly 전처리 적용 (YOLO 모델용)
        if apply_preprocessing_on_the_fly:
            # apply_custom_preprocessing는 모델 입력에 적합한 float32, [0,1] 범위의 이미지를 반환 (use_normalization=True 기본값)
            img_for_prediction = apply_custom_preprocessing(img, use_clahe=True, use_noise_reduction=True, use_normalization=True)
        else:
            # 전처리를 적용하지 않는 경우, YOLO 모델은 일반적으로 uint8, BGR, [0,255] 이미지를 예상함
            # Ultralytics YOLO 내부에서 정규화 등을 처리.
            img_for_prediction = img 

        # 이미지 크기 (YOLO 좌표 정규화에 사용)
        # 원본 이미지 크기를 사용해야 함. 전처리로 크기가 변경되지 않았다고 가정.
        # 만약 apply_custom_preprocessing가 크기를 변경한다면, 그에 맞게 h, w를 가져와야 함.
        # 현재 apply_custom_preprocessing는 크기를 변경하지 않음.
        h, w, _ = img.shape # 원본 이미지 기준으로 크기 계산

        # YOLO 모델로 객체 감지 (전처리된 또는 원본 이미지 사용)
        results = yolo_model.predict(img_for_prediction, conf_thresh=0.25)

        # 결과를 YOLO 형식으로 변환
        lines = []
        for box in results:
            # 박스 정보 추출 (x1, y1, x2, y2, 신뢰도, 클래스 ID)
            if len(box) >= 6:  # 클래스 ID가 포함된 경우
                x1, y1, x2, y2, sc, cls_id = box[:6]
            else:  # 클래스 ID가 없는 경우 (기본값: 0)
                x1, y1, x2, y2, sc = box[:5]
                cls_id = 0  # 기본값: 경차/세단

            # 픽셀 좌표를 YOLO 형식(중심점, 너비, 높이, 정규화)으로 변환
            bw = x2 - x1  # 박스 너비
            bh = y2 - y1  # 박스 높이
            x_ctr = x1 + bw / 2  # 중심 X
            y_ctr = y1 + bh / 2  # 중심 Y

            # 이미지 크기로 정규화 (0~1 범위)
            x_ctr /= w
            y_ctr /= h
            bw /= w
            bh /= h

            # YOLO 형식 라인 생성: "class_id x_center y_center width height score"
            line = f"{int(cls_id)} {x_ctr:.6f} {y_ctr:.6f} {bw:.6f} {bh:.6f} {sc:.3f}"
            lines.append(line)

        # 결과를 텍스트 파일로 저장
        with open(out_txt, "w") as fw:
            for ln in lines:
                fw.write(ln + "\n")

    print(
        f"[inference_yolo] 추론 완료: {len(files)}개 이미지, 결과 저장 경로: {out_label_dir}"
    )
    return out_label_dir


def inference_dfine(dfine_model, test_dir, out_label_dir=None, apply_preprocessing_on_the_fly=False):
    """
    D-FINE 모델을 사용하여 테스트 이미지에서 객체 감지 추론을 수행합니다.

    매개변수:
        dfine_model: D-FINE 모델 인스턴스
        test_dir: 테스트 이미지 디렉토리 경로
        out_label_dir: 출력 라벨 디렉토리 경로 (기본값: test_dir/labels_pred)
        apply_preprocessing_on_the_fly (bool): 추론 시점에 커스텀 전처리 적용 여부

    반환값:
        출력 라벨 디렉토리 경로
    """
    if out_label_dir is None:
        out_label_dir = os.path.join(test_dir, "labels_pred")

    os.makedirs(out_label_dir, exist_ok=True)
    img_dir = os.path.join(test_dir, "images")
    files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg"))]
    )

    print(f"[inference_dfine] 테스트 이미지: {len(files)}개")

    for f in files:
        # 이미지 경로 및 출력 파일 경로 생성
        imgp = os.path.join(img_dir, f)
        base = os.path.splitext(f)[0]
        out_txt = os.path.join(out_label_dir, base + ".txt")

        # 한글 경로 지원을 위한 이미지 로딩
        try:
            with open(imgp, 'rb') as img_file:
                img_bytes = bytearray(img_file.read())
                img_array = np.asarray(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[경고] 이미지를 읽을 수 없습니다: {imgp} (오류: {e})")
            continue

        if img is None:
            print(f"[경고] 이미지를 디코딩할 수 없습니다: {imgp}")
            continue

        # On-the-fly 전처리 적용 (D-FINE 모델용)
        img_for_prediction = img # 기본적으로 원본 이미지 사용
        if apply_preprocessing_on_the_fly:
            # D-FINE 모델이 어떤 입력을 기대하는지에 따라 use_normalization 등을 조정해야 할 수 있음.
            # 여기서는 YOLO와 동일하게 float32, [0,1]로 정규화된 이미지를 사용한다고 가정.
            img_for_prediction = apply_custom_preprocessing(img, use_clahe=True, use_noise_reduction=True, use_normalization=True)
        
        # 이미지 크기 (YOLO 좌표 정규화에 사용)
        # 원본 이미지 크기 사용
        h, w, _ = img.shape

        # BGR -> RGB로 변환
        # img_for_prediction은 전처리 후에도 BGR 상태일 것임 (apply_custom_preprocessing가 BGR 반환)
        processed_img_rgb = cv2.cvtColor(img_for_prediction, cv2.COLOR_BGR2RGB)
        
        # 이미지를 텐서로 변환
        # D-FINE 모델이 float32, [0,1] 입력을 받는다고 가정.
        # 만약 img_for_prediction이 apply_custom_preprocessing(use_normalization=True)를 거쳤다면 이미 float32, [0,1]
        # 그렇지 않다면 (uint8, 0-255), 여기서 정규화 필요.
        if img_for_prediction.dtype == np.uint8: # 즉, apply_preprocessing_on_the_fly=False였거나, True여도 use_normalization=False인 경우
             img_tensor = torch.from_numpy(processed_img_rgb.transpose(2, 0, 1)).float() / 255.0 
        else: # 이미 float32, [0,1] 상태라고 가정
             img_tensor = torch.from_numpy(processed_img_rgb.transpose(2, 0, 1))

        # D-FINE 모델로 객체 감지
        results = dfine_model.predict(img_tensor, conf_thr=0.25)

        # 결과를 YOLO 형식으로 변환
        lines = []
        for box in results:
            # 박스 정보 추출 (x1, y1, x2, y2, 신뢰도, 클래스 ID)
            x1, y1, x2, y2, sc, cls_id = box

            # 픽셀 좌표를 YOLO 형식(중심점, 너비, 높이, 정규화)으로 변환
            bw = x2 - x1  # 박스 너비
            bh = y2 - y1  # 박스 높이
            x_ctr = x1 + bw / 2  # 중심 X
            y_ctr = y1 + bh / 2  # 중심 Y

            # 이미지 크기로 정규화 (0~1 범위)
            x_ctr /= w
            y_ctr /= h
            bw /= w
            bh /= h

            # YOLO 형식 라인 생성: "class_id x_center y_center width height score"
            line = f"{int(cls_id)} {x_ctr:.6f} {y_ctr:.6f} {bw:.6f} {bh:.6f} {sc:.3f}"
            lines.append(line)

        # 결과를 텍스트 파일로 저장
        with open(out_txt, "w") as fw:
            for ln in lines:
                fw.write(ln + "\n")

    print(
        f"[inference_dfine] 추론 완료: {len(files)}개 이미지, 결과 저장 경로: {out_label_dir}"
    )
    return out_label_dir


def generate_prediction(model, test_dir, out_label_dir=None, preprocess_on_the_fly=False):
    """
    지정된 모델을 사용하여 테스트 이미지에서 객체를 감지하고 결과를 YOLO 형식으로 저장합니다.

    매개변수:
        model: 학습된 모델 객체 (YOLOModel 또는 DFineModel)
        test_dir (str): 테스트 이미지가 있는 디렉토리 경로
        out_label_dir (str, optional): 결과 라벨을 저장할 디렉토리 경로
        preprocess_on_the_fly (bool): 추론 시점에 커스텀 전처리 적용 여부

    반환값:
        str: 결과 라벨 디렉토리 경로
    """
    # 모델 유형에 따라 추론 함수 선택
    from models.yolo_v8 import YOLOModel
    from models.dfine_b import DFineModel
    
    if isinstance(model, YOLOModel):
        lbl_dir = inference_yolo(model, test_dir, out_label_dir, apply_preprocessing_on_the_fly=preprocess_on_the_fly)
    elif isinstance(model, DFineModel):
        lbl_dir = inference_dfine(model, test_dir, out_label_dir, apply_preprocessing_on_the_fly=preprocess_on_the_fly)
    else:
        raise ValueError(f"지원되지 않는 모델 유형: {type(model)}")
        
    return lbl_dir
