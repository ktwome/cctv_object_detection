import os
import argparse

# YOLO 라벨 포맷: class x_center y_center w h (공백 구분, 최소 5개 항목)
def check_label_format(label_path):
    try:
        with open(label_path, encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                return False
            # 클래스 번호가 int, 나머지는 float로 변환 가능한지 체크
            int(parts[0])
            [float(x) for x in parts[1:5]]
        return True
    except Exception:
        return False

def main(images_dir, labels_dir):
    images = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    images_set = set(os.path.splitext(f)[0] for f in images)
    labels_set = set(os.path.splitext(f)[0] for f in labels)

    print(f"[요약] 이미지 수: {len(images)}, 라벨 수: {len(labels)}")

    # 1. 라벨 없는 이미지
    no_label_imgs = images_set - labels_set
    if no_label_imgs:
        print(f"[경고] 라벨 없는 이미지: {len(no_label_imgs)}개")
        for name in sorted(no_label_imgs)[:10]:
            print(f"  - {name}")
    else:
        print("[OK] 모든 이미지에 라벨이 있습니다.")

    # 2. 이미지 없는 라벨
    no_img_labels = labels_set - images_set
    if no_img_labels:
        print(f"[경고] 이미지 없는 라벨: {len(no_img_labels)}개")
        for name in sorted(no_img_labels)[:10]:
            print(f"  - {name}")
    else:
        print("[OK] 모든 라벨이 이미지와 매칭됩니다.")

    # 3. 0바이트 라벨
    zero_byte_labels = [f for f in labels if os.path.getsize(os.path.join(labels_dir, f)) == 0]
    if zero_byte_labels:
        print(f"[경고] 0바이트 라벨 파일: {len(zero_byte_labels)}개")
        for f in zero_byte_labels[:10]:
            print(f"  - {f}")
    else:
        print("[OK] 0바이트 라벨 파일 없음.")

    # 4. 포맷 오류 라벨
    format_error_labels = [f for f in labels if os.path.getsize(os.path.join(labels_dir, f)) > 0 and not check_label_format(os.path.join(labels_dir, f))]
    if format_error_labels:
        print(f"[경고] 포맷 오류 라벨 파일: {len(format_error_labels)}개")
        for f in format_error_labels[:10]:
            print(f"  - {f}")
    else:
        print("[OK] 모든 라벨 파일 포맷 정상.")

    print("\n[점검 완료]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 데이터셋 이미지-라벨 매칭 및 라벨 유효성 점검 스크립트")
    parser.add_argument('--images', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--labels', type=str, required=True, help='라벨 폴더 경로')
    args = parser.parse_args()
    main(args.images, args.labels) 