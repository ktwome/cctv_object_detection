'''
extractor.py 사용법
1. https://aihub.or.kr/aihubdata/data/dwld.do?currMenu=115&topMenu=100 에서 데이터를 다운로드 받는다.
2. 다운로드 받은 데이터를 압축 해제한다.
3. 압축 해제된 디렉토리에 extractor.py를 넣는다.
4. 터미널에서 extractor.py를 실행한다.
'''

import os
import zipfile
import shutil
from pathlib import Path
import glob
import subprocess
import sys
import platform
import locale

# 기본 설정
base_dir = Path('.')
data_categories = {
    '1.Training': 'train',
    '2.Validation': 'val',
    '3.Test': 'test'
}

# 한글 인코딩 확인 및 설정
print("=== 시스템 정보 ===")
print(f"OS: {platform.system()} {platform.version()}")
print(f"Python: {sys.version}")
print(f"기본 인코딩: {sys.getfilesystemencoding()}")
print(f"현재 로케일: {locale.getpreferredencoding()}")

def create_directory(directory):
    """디렉토리 생성 함수"""
    os.makedirs(directory, exist_ok=True)
    print(f"디렉토리 생성/확인 완료: {directory}")

def extract_zip_with_external_tool(zip_path, extract_to):
    """외부 도구(7zip 또는 PowerShell)를 사용하여 ZIP 파일 압축 해제 함수"""
    try:
        zip_path_str = str(zip_path)
        extract_to_str = str(extract_to)
        
        if platform.system() == 'Windows':
            # 7zip이 있는지 확인
            seven_zip_path = r"C:\Program Files\7-Zip\7z.exe"
            if os.path.exists(seven_zip_path):
                # 7zip 사용
                cmd = [seven_zip_path, 'x', zip_path_str, f'-o{extract_to_str}', '-y']
                subprocess.run(cmd, check=True)
                print(f"7-Zip으로 압축 해제 완료: {zip_path}")
            else:
                # PowerShell 사용
                ps_cmd = f'Expand-Archive -Path "{zip_path_str}" -DestinationPath "{extract_to_str}" -Force'
                cmd = ['powershell', '-command', ps_cmd]
                subprocess.run(cmd, check=True)
                print(f"PowerShell로 압축 해제 완료: {zip_path}")
        else:
            # 다른 OS (Linux, macOS)에서는 기본 unzip 명령어 사용
            cmd = ['unzip', '-o', zip_path_str, '-d', extract_to_str]
            subprocess.run(cmd, check=True)
            print(f"unzip으로 압축 해제 완료: {zip_path}")
        
        return True
    except Exception as e:
        print(f"압축 해제 실패: {zip_path}, 오류: {e}")
        # 내장 zipfile 모듈 사용 시도
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # CP949 인코딩 문제 해결 위해 파일 이름 변환
                for file_info in zip_ref.infolist():
                    try:
                        # 한글 파일명 처리
                        if platform.system() == 'Windows':
                            file_info.filename = file_info.filename.encode('cp437').decode('cp949')
                        else:
                            file_info.filename = file_info.filename.encode('cp437').decode('utf-8')
                    except:
                        pass  # 변환 실패 시 원래 이름 사용
                    
                    try:
                        zip_ref.extract(file_info, extract_to)
                    except Exception as extract_error:
                        print(f"  - 파일 추출 실패: {file_info.filename}, 오류: {extract_error}")
            
            print(f"내장 zipfile로 압축 해제 완료: {zip_path}")
            return True
        except Exception as zip_error:
            print(f"모든 압축 해제 방법 실패: {zip_path}, 오류: {zip_error}")
            return False

def move_files(source_dir, target_dir, file_patterns):
    """특정 패턴의 파일들을 이동시키는 함수"""
    count = 0
    for pattern in file_patterns:
        try:
            for file_path in glob.glob(os.path.join(source_dir, '**', pattern), recursive=True):
                if os.path.isfile(file_path):
                    file_name = os.path.basename(file_path)
                    target_path = os.path.join(target_dir, file_name)
                    shutil.copy2(file_path, target_path)
                    count += 1
        except Exception as e:
            print(f"파일 복사 오류 ({pattern}): {e}")
    
    return count

def process_category(category_source, target_name):
    """각 카테고리(Training, Validation, Test) 처리 함수"""
    print(f"\n=== {category_source} 처리 시작 ({target_name}) ===")
    
    # 결과 디렉토리 생성
    target_dir = base_dir / target_name
    images_dir = target_dir / 'images'
    labels_dir = target_dir / 'labels_json'
    
    create_directory(target_dir)
    create_directory(images_dir)
    create_directory(labels_dir)
    
    # 임시 디렉토리 생성
    temp_source = target_dir / 'temp_source'
    temp_label = target_dir / 'temp_label'
    create_directory(temp_source)
    create_directory(temp_label)
    
    # 원천데이터 및 라벨링데이터 경로
    source_data_dir = base_dir / category_source / '원천데이터'
    label_data_dir = base_dir / category_source / '라벨링데이터'
    
    # 모든 하위 디렉토리에서 zip 파일 찾기
    print(f"원천데이터 압축 파일 찾는 중...")
    source_zips = list(source_data_dir.glob('**/*.zip'))
    print(f"총 {len(source_zips)}개 원천데이터 압축 파일 발견")
    
    for zip_file in source_zips:
        if zip_file.exists():
            print(f"처리 중: {zip_file}")
            extract_zip_with_external_tool(zip_file, temp_source)
    
    print(f"라벨링데이터 압축 파일 찾는 중...")
    label_zips = list(label_data_dir.glob('**/*.zip'))
    print(f"총 {len(label_zips)}개 라벨링데이터 압축 파일 발견")
    
    for zip_file in label_zips:
        if zip_file.exists():
            print(f"처리 중: {zip_file}")
            extract_zip_with_external_tool(zip_file, temp_label)
    
    # 이미지 파일 이동
    print(f"이미지 파일 이동 중...")
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
    img_count = move_files(temp_source, images_dir, image_patterns)
    print(f"총 {img_count}개의 이미지 파일을 '{images_dir}' 폴더로 이동했습니다.")
    
    # JSON 파일 이동
    print(f"JSON 파일 이동 중...")
    json_patterns = ['*.json']
    json_count = move_files(temp_label, labels_dir, json_patterns)
    print(f"총 {json_count}개의 JSON 파일을 '{labels_dir}' 폴더로 이동했습니다.")
    
    # 임시 디렉토리 정리
    print(f"임시 파일 정리 중...")
    shutil.rmtree(temp_source, ignore_errors=True)
    shutil.rmtree(temp_label, ignore_errors=True)
    
    print(f"=== {category_source} 처리 완료 ===")
    return img_count, json_count

def process_sample():
    """Sample 폴더 처리 함수"""
    print("\n=== 4.Sample 처리 시작 ===")
    
    sample_zip = base_dir / '4.Sample' / 'Sample.zip'
    sample_dir = base_dir / 'sample'
    
    create_directory(sample_dir)
    
    if sample_zip.exists():
        extract_zip_with_external_tool(sample_zip, sample_dir)
        print(f"Sample 압축 해제 완료: {sample_dir}")
    else:
        print(f"Sample 압축 파일이 존재하지 않습니다: {sample_zip}")
    
    print("=== 4.Sample 처리 완료 ===")

# 메인 실행
def main():
    print("=== 데이터 추출 및 정리 시작 ===")
    total_images = 0
    total_json = 0
    
    # 각 카테고리 처리
    for category, target in data_categories.items():
        if (base_dir / category).exists():
            img_count, json_count = process_category(category, target)
            total_images += img_count
            total_json += json_count
        else:
            print(f"\n!!! {category} 디렉토리가 존재하지 않습니다 !!!")
    
    # Sample 처리
    if (base_dir / '4.Sample').exists():
        process_sample()
    
    print("\n=== 작업 최종 완료! ===")
    print(f"총 {total_images}개의 이미지 파일과 {total_json}개의 JSON 파일이 처리되었습니다.")
    print("폴더 구조:")
    print("- train/")
    print("  |- images/")
    print("  |- labels_json/")
    print("- val/")
    print("  |- images/")
    print("  |- labels_json/")
    print("- test/")
    print("  |- images/")
    print("  |- labels_json/")
    print("- sample/")

if __name__ == "__main__":
    main() 