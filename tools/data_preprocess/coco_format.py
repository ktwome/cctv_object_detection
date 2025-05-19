import cv2
import json
import os
import datetime
import argparse
from tqdm import tqdm
import numpy as np  # NumPy 모듈 추가


def create_coco_labels(img_dir, json_dir, out_dir, phase="train", overwrite=False):
    """
    JSON 형식의 어노테이션을 COCO 형식 라벨 파일로 변환합니다.
    
    매개변수:
        img_dir (str): 이미지가 저장된 디렉토리 경로
        json_dir (str): JSON 어노테이션이 저장된 디렉토리 경로
        out_dir (str): COCO 형식 JSON 파일이 저장될 디렉토리 경로
        phase (str): 데이터셋 단계(train, val, test)
        overwrite (bool): 이미 존재하는 파일을 덮어쓸지 여부
    
    반환:
        int: 처리된 이미지 수
    """
    # 출력 파일 경로 생성
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{phase}.json")
    
    # 이미 파일이 존재하고 덮어쓰기 옵션이 False인 경우 스킵
    if os.path.exists(out_file) and not overwrite:
        print(f"[create_coco_labels] 출력 파일이 이미 존재합니다: {out_file}")
        print(f"덮어쓰려면 overwrite=True 옵션을 사용하세요.")
        return 0

    # 클래스 매핑 사전 정의 (data_preprocessing.py 참조)
    class_mapping = {
        "경차/세단": 0,
        "SUV/승합차": 1,
        "트럭": 2,
        "버스(소형, 대형)": 3,
        "통학버스(소형,대형)": 4,
        "경찰차": 5,
        "구급차": 6,
        "소방차": 7,
        "견인차": 8,
        "기타 특장차": 9,
        "성인": 10,
        "어린이": 11,
        "오토바이": 12,
        "자전거 / 기타 전동 이동체": 13,
        "라바콘": 14,
        "삼각대": 15,
        "기타": 16,
    }

    # 디렉토리 존재 확인
    if not os.path.exists(img_dir):
        print(f"[create_coco_labels] 오류: 이미지 디렉토리가 존재하지 않음: {img_dir}")
        return 0

    if not os.path.exists(json_dir):
        print(f"[create_coco_labels] 오류: JSON 디렉토리가 존재하지 않음: {json_dir}")
        return 0

    print(
        f"[create_coco_labels] 디렉토리 확인 완료 - 이미지: {img_dir}, JSON: {json_dir}, 출력: {out_dir}"
    )
    
    # 이미지 및 JSON 파일 목록 가져오기
    img_files = {}
    for f in os.listdir(img_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            # 파일명에서 확장자 제거
            base_name = os.path.splitext(f)[0]
            img_files[base_name] = f

    json_files = {}
    for f in os.listdir(json_dir):
        if f.lower().endswith(".json"):
            # JSON 파일명에서 가능한 '.jpg' 부분과 '.json' 확장자 모두 제거
            base_name = f.replace(".jpg.json", "").replace(".json", "")
            json_files[base_name] = f

    print(
        f"[create_coco_labels] 발견된 파일 - 이미지: {len(img_files)}개, JSON: {len(json_files)}개"
    )

    # 공통 기본 이름 찾기
    common_names = set(img_files.keys()) & set(json_files.keys())

    if not common_names:
        print("[create_coco_labels] 오류: 매칭되는 이미지-JSON 파일 쌍이 없음")
        print(f"이미지 파일 예시: {list(img_files.keys())[:3]}")
        print(f"JSON 파일 예시: {list(json_files.keys())[:3]}")
        return 0

    print(f"[create_coco_labels] 매칭된 파일 쌍: {len(common_names)}개")

    # COCO 포맷 초기화
    coco_output = {
        "info": {
            "description": "CCTV 객체 감지 데이터셋",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "Ktwome",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial",
                "url": "http://creativecommons.org/licenses/by-nc/2.0/"
            }
        ],
        "categories": [
            {"id": id, "name": name, "supercategory": "object"} 
            for name, id in class_mapping.items()
        ],
        "images": [],
        "annotations": []
    }

    processed_count = 0
    skipped_count = 0
    error_count = 0
    class_counts = {i: 0 for i in range(len(class_mapping))}
    annotation_id = 1  # COCO 형식에서는 각 어노테이션마다 고유 ID 필요

    # 각 이미지와 어노테이션 처리
    for image_id, base_name in enumerate(tqdm(sorted(common_names), desc="[create_coco_labels] 이미지 처리 중")):
        image_id += 1  # COCO에서는 image_id가 1부터 시작
        
        try:
            # 이미지 파일 로드하여 크기 확인
            img_path = os.path.join(img_dir, img_files[base_name])
            
            # 윈도우 경로를 POSIX 형식으로 정규화 (백슬래시 -> 슬래시)
            normalized_path = img_path.replace('\\', '/')
            
            # 한글 경로 문제 해결을 위해 NumPy를 이용하여 파일 직접 로드
            try:
                with open(normalized_path, 'rb') as file:
                    img_array = bytearray(file.read())
                    img = cv2.imdecode(np.frombuffer(img_array, np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[create_coco_labels] 경고: 이미지를 로드할 수 없음: {normalized_path}, 오류: {str(e)}")
                error_count += 1
                continue
                
            if img is None:
                print(f"[create_coco_labels] 경고: 이미지를 로드할 수 없음: {normalized_path}")
                error_count += 1
                continue

            img_h, img_w = img.shape[:2]
            
            # 이미지 정보 추가
            coco_output["images"].append({
                "id": image_id,
                "file_name": img_files[base_name],
                "width": img_w,
                "height": img_h,
                "license": 1,
                "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # JSON 파일 로드
            json_path = os.path.join(json_dir, json_files[base_name])
            # 윈도우 경로를 POSIX 형식으로 정규화
            json_path = json_path.replace('\\', '/')
            
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            except json.JSONDecodeError:
                print(f"[create_coco_labels] 경고: JSON 파싱 오류: {json_path}")
                error_count += 1
                continue

            # 새로운 JSON 형식에 맞게 처리
            if "row" not in json_data:
                print(f"[create_coco_labels] 경고: JSON에 'row' 키가 없음: {json_path}")
                error_count += 1
                continue

            # 각 객체 어노테이션 처리
            for obj in json_data["row"]:
                try:
                    # 객체 유형 및 클래스 ID 추출
                    obj_type = obj["attributes2"]
                    
                    # 클래스 ID 결정
                    class_id = 16  # 기본값: "기타"
                    if obj_type in class_mapping:
                        class_id = class_mapping[obj_type]

                    # 좌표 처리 (points1~points4에서 바운딩 박스 추출)
                    points = []
                    for i in range(1, 5):
                        key = f"points{i}"
                        if key in obj and obj[key]:
                            try:
                                x, y = map(float, obj[key].split(","))
                                points.append((x, y))
                            except (ValueError, AttributeError):
                                continue

                    # 4개의 점이 없다면 계속
                    if len(points) != 4:
                        print(f"[경고] 객체 좌표가 부족합니다: {json_path}")
                        continue

                    # 바운딩 박스 계산 (COCO 형식: [x_min, y_min, width, height])
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x_min, x_max = max(0, min(xs)), min(img_w, max(xs))
                    y_min, y_max = max(0, min(ys)), min(img_h, max(ys))
                    
                    # 너비와 높이 계산
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # 유효성 검증
                    if width <= 0 or height <= 0:
                        print(f"[경고] 유효하지 않은 바운딩 박스 크기: {width}x{height}")
                        continue

                    # COCO 형식 어노테이션 추가
                    coco_output["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, width, height],
                        "area": width * height,
                        "segmentation": [],  # 세그멘테이션 정보는 없음
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1
                    class_counts[class_id] += 1

                except Exception as e:
                    print(f"[create_coco_labels] 객체 처리 중 오류: {str(e)}")
                    continue

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"[create_coco_labels] 진행 상황: {processed_count}개 처리 중...")
            
        except Exception as e:
            print(f"[create_coco_labels] 예외 발생 ({base_name}): {str(e)}")
            error_count += 1
            continue

    # COCO 파일 저장
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=2)

    print(
        f"[create_coco_labels] 처리 완료: {processed_count}개 처리, {skipped_count}개 생략, {error_count}개 오류"
    )
    print(f"[create_coco_labels] COCO 형식 어노테이션 저장 완료: {out_file}")
    print(f"총 이미지 수: {len(coco_output['images'])}, 총 어노테이션 수: {len(coco_output['annotations'])}")

    # 클래스별 통계 출력
    print(f"[create_coco_labels] 클래스별 객체 수:")
    for class_id, count in class_counts.items():
        if count > 0:
            class_name = next(
                (name for name, cid in class_mapping.items() if cid == class_id),
                f"클래스 {class_id}",
            )
            print(f"  - {class_name}: {count}개")

    if processed_count == 0:
        print(
            "[create_coco_labels] 주의: 아무 파일도 처리되지 않았습니다. 경로와 파일 형식을 확인하세요."
        )

    return processed_count


def process_all_datasets(base_dir="data", overwrite=False):
    """
    train, val, test 세 가지 데이터셋에 대해 COCO 형식 라벨 변환을 수행합니다.
    
    매개변수:
        base_dir (str): 데이터 베이스 디렉토리 경로
        overwrite (bool): 기존 파일 덮어쓰기 여부
    """
    phases = ["train", "val", "test"]
    total_processed = 0
    
    for phase in phases:
        print(f"\n===== {phase.upper()} 데이터셋 COCO 라벨 변환 시작 =====")
        img_dir = os.path.join(base_dir, phase, "images")
        json_dir = os.path.join(base_dir, phase, "labels_json")
        out_dir = os.path.join(base_dir, phase, "labels_coco")
        
        # 디렉토리가 존재하는지 확인
        if not os.path.exists(img_dir):
            print(f"[process_all_datasets] 경고: {img_dir} 디렉토리가 존재하지 않습니다. {phase} 단계 건너뜁니다.")
            continue
            
        if not os.path.exists(json_dir):
            print(f"[process_all_datasets] 경고: {json_dir} 디렉토리가 존재하지 않습니다. {phase} 단계 건너뜁니다.")
            continue
        
        # 출력 디렉토리 생성
        os.makedirs(out_dir, exist_ok=True)
        
        # COCO 라벨 생성 실행
        processed = create_coco_labels(img_dir, json_dir, out_dir, phase=phase, overwrite=overwrite)
        total_processed += processed
        print(f"===== {phase.upper()} 데이터셋 처리 완료: {processed}개 파일 변환됨 =====")
    
    print(f"\n모든 데이터셋 처리 완료! 총 {total_processed}개 파일 변환됨")
    return total_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON 어노테이션을 COCO 형식으로 변환")
    parser.add_argument("--base_dir", default="data", help="데이터 베이스 디렉토리 경로")
    parser.add_argument("--img_dir", help="이미지 디렉토리 경로 (단일 데이터셋 처리 시)")
    parser.add_argument("--json_dir", help="JSON 어노테이션 디렉토리 경로 (단일 데이터셋 처리 시)")
    parser.add_argument("--out_dir", help="COCO 형식 라벨 출력 디렉토리 (단일 데이터셋 처리 시)")
    parser.add_argument("--phase", default="train", help="데이터셋 단계(train, val, test) (단일 데이터셋 처리 시)")
    parser.add_argument("--overwrite", action="store_true", help="기존 파일 덮어쓰기")
    parser.add_argument("--all", action="store_true", help="모든 데이터셋(train, val, test) 처리")
    
    args = parser.parse_args()
    
    # 모든 데이터셋 처리 모드
    if args.all or (args.img_dir is None and args.json_dir is None and args.out_dir is None):
        process_all_datasets(args.base_dir, args.overwrite)
    # 단일 데이터셋 처리 모드
    elif args.img_dir and args.json_dir and args.out_dir:
        # COCO 변환 실행
        create_coco_labels(
            args.img_dir,
            args.json_dir,
            args.out_dir,
            phase=args.phase,
            overwrite=args.overwrite
        )
    else:
        print("오류: 단일 데이터셋 처리 시 --img_dir, --json_dir, --out_dir가 모두 필요합니다.")
        print("또는 --all 옵션을 사용하여 모든 데이터셋을 처리하세요.")
        parser.print_help() 