from PIL import Image
import os
import cv2
import numpy as np
import argparse
import yaml
import io

PROJECT_ROOT = "D:/CDM/Workspace/IoT/cctv_object_detection"

def load_class_names(data_yaml_path):
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if 'names' in data:
                if isinstance(data['names'], list):
                    return data['names']
                elif isinstance(data['names'], dict):
                    sorted_keys = sorted(data['names'].keys(), key=lambda x: int(x))
                    return [data['names'][k] for k in sorted_keys]
        except Exception as e:
            print(f"data.yaml 오류: {e}")
    return None

def visualize_dataset(image_dir, output_dir, class_names=None):
    abs_image_dir = os.path.join(PROJECT_ROOT, image_dir) if not os.path.isabs(image_dir) else image_dir
    abs_output_dir = os.path.join(PROJECT_ROOT, output_dir) if not os.path.isabs(output_dir) else output_dir
    abs_image_dir = os.path.normpath(abs_image_dir)
    abs_output_dir = os.path.normpath(abs_output_dir)

    parent_dir = os.path.dirname(abs_image_dir)
    abs_label_dir = os.path.join(parent_dir, "labels")
    abs_label_dir = os.path.normpath(abs_label_dir)

    if not os.path.exists(abs_label_dir):
        print(f"라벨 디렉토리 없음: {abs_label_dir}")
        return

    os.makedirs(abs_output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(abs_image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    for img_file in image_files:
        abs_image_path = os.path.normpath(os.path.join(abs_image_dir, img_file))
        abs_label_path = os.path.normpath(os.path.join(abs_label_dir, os.path.splitext(img_file)[0] + ".txt"))

        try:
            with open(abs_image_path, 'rb') as f:
                img_bytes = f.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = np.array(pil_img)
        except Exception as e:
            print(f"이미지 로드 실패: {abs_image_path}, 오류: {e}")
            continue

        h, w = img.shape[:2]

        if not os.path.exists(abs_label_path):
            cv2.imwrite(os.path.join(abs_output_dir, img_file), img)
            continue

        try:
            with open(abs_label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"라벨 로딩 실패: {abs_label_path}, 오류: {e}")
            continue

        for line in lines:
            try:
                class_id, cx, cy, bw, bh = map(float, line.strip().split())
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                label = class_names[int(class_id)] if class_names and int(class_id) < len(class_names) else str(int(class_id))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except Exception as e:
                print(f"라벨 처리 오류: {abs_label_path}, 줄: {line.strip()}, 오류: {e}")

        cv2.imwrite(os.path.join(abs_output_dir, img_file), img)

    print("시각화 완료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, help="이미지 디렉토리")
    parser.add_argument("--output_dir", type=str, default="visualized_results")
    args = parser.parse_args()

    abs_image_input_dir = os.path.join(PROJECT_ROOT, args.image_dir) if not os.path.isabs(args.image_dir) else args.image_dir
    data_yaml_dir = os.path.dirname(abs_image_input_dir)
    data_yaml_path_abs = os.path.join(data_yaml_dir, "custom_data.yaml")

    class_names_list = load_class_names(data_yaml_path_abs)
    visualize_dataset(args.image_dir, args.output_dir, class_names=class_names_list)