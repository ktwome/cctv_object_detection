import cv2
import os
import glob
import argparse
from tqdm import tqdm


def basic_image_preprocess(
    input_dir,
    output_dir,
    use_gray=True,
    use_clahe=True,
    overwrite=False,
):
    """
    CCTV 이미지에 대한 기본적인 전처리를 수행합니다.

    매개변수:
        input_dir (str): 원본 이미지가 저장된 디렉토리 경로
        output_dir (str): 전처리된 이미지를 저장할 디렉토리 경로
        use_gray (bool): 그레이스케일 변환 여부
        use_clahe (bool): CLAHE(대비 제한 적응형 히스토그램 평활화) 적용 여부
        overwrite (bool): 이미 존재하는 파일을 덮어쓸지 여부
            - True: 모든 이미지를 다시 처리
            - False: 이미 처리된 이미지는 건너뜀 (기본값)

    동작:
        1. 입력 디렉토리에서 모든 JPG 이미지를 로드
        2. 필요시 그레이스케일 변환 및 CLAHE 적용
        3. 결과 이미지를 출력 디렉토리에 저장
    """
    os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리 생성 (없으면)
    img_files = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg"))
    )  # 모든 JPG 파일 가져오기

    processed_count = 0
    skipped_count = 0

    for f in tqdm(img_files, desc="[basic_image_preprocess] 이미지 처리 중"):
        # 이미지 로드
        img = cv2.imread(f)
        if img is None:
            print(f"[경고] 이미지 읽기 실패: {f}")
            continue

        # 그레이스케일 변환 (선택 사항)
        if use_gray:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # CLAHE 적용 (선택 사항)
        if use_clahe:
            # YCrCb 색상 공간으로 변환 (Y: 밝기 채널)
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            # 밝기 채널에만 CLAHE 적용
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y = clahe.apply(y)
            # 채널 합치기
            merged = cv2.merge((y, cr, cb))
            img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        # 결과 이미지 저장
        base = os.path.basename(f)
        out_path = os.path.join(output_dir, base)

        # overwrite=False이고 이미 파일이 존재하면 건너뜀
        if not overwrite and os.path.exists(out_path):
            skipped_count += 1
            continue

        cv2.imwrite(out_path, img)
        processed_count += 1

    print(
        f"[basic_image_preprocess] 전처리 완료 => {output_dir}, 처리된 이미지 수: {processed_count}, 생략된 이미지 수: {skipped_count}"
    )
    return processed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV 이미지 전처리 도구")
    parser.add_argument("--input_dir", required=True, help="입력 이미지 디렉토리")
    parser.add_argument("--output_dir", required=True, help="출력 이미지 디렉토리")
    parser.add_argument("--use_gray", action="store_true", help="그레이스케일 변환 적용")
    parser.add_argument("--use_clahe", action="store_true", help="CLAHE 적용")
    parser.add_argument("--overwrite", action="store_true", help="기존 파일 덮어쓰기")
    
    args = parser.parse_args()
    
    # 이미지 전처리 실행
    basic_image_preprocess(
        args.input_dir,
        args.output_dir,
        use_gray=args.use_gray,
        use_clahe=args.use_clahe,
        overwrite=args.overwrite
    ) 