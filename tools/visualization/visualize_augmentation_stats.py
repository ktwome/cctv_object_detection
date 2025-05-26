#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터 증강 결과 시각화 스크립트
matplotlib을 사용하여 증강된 이미지의 분포 및 유형별 통계를 시각화합니다.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
import traceback
import json # json 모듈 추가

# 한글 폰트 설정 (Windows 기준, 다른 OS의 경우 폰트 경로 확인 필요)
# 사용 가능한 한글 폰트 경로를 지정해주세요.
# 예: 'Malgun Gothic' (Windows), 'AppleGothic' (macOS)
# 경로를 직접 지정할 수도 있습니다. 예: fm.FontProperties(fname="path/to/your/font.ttf")
DEFAULT_FONT_NAME = 'Malgun Gothic'

def get_korean_font():
    """사용 가능한 한글 폰트를 반환하거나 기본 폰트를 사용합니다."""
    try:
        # 시스템에서 사용 가능한 모든 폰트 목록 가져오기 (시간이 다소 걸릴 수 있음)
        # font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        # malgun_fonts = [f for f in font_list if 'malgun' in f.lower()]
        # if malgun_fonts:
        #     font_path = malgun_fonts[0]
        #     font_prop = fm.FontProperties(fname=font_path)
        #     plt.rcParams['font.family'] = font_prop.get_name()
        #     print(f"한글 폰트 '{font_prop.get_name()}' ({font_path})로 설정되었습니다.")
        # else:
        #     raise Exception("Malgun Gothic 폰트를 찾을 수 없습니다.")
        
        # 특정 폰트 이름으로 직접 시도
        font_prop = fm.FontProperties(family=DEFAULT_FONT_NAME)
        plt.rcParams['font.family'] = font_prop.get_name() # 이 부분이 실제 폰트를 적용
        fm.findfont(font_prop) # 폰트 존재 여부 확인 (없으면 오류 발생하여 except로 감)

        plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
        print(f"한글 폰트 '{font_prop.get_name()}'로 설정 시도.")
        # 실제로 적용되었는지 테스트하기 위해 샘플 텍스트 출력
        # fig, ax = plt.subplots()
        # ax.text(0.5, 0.5, "한글 테스트", fontproperties=font_prop)
        # plt.close(fig)
        return font_prop
    except Exception as e:
        print(f"경고: 지정된 한글 폰트 '{DEFAULT_FONT_NAME}'를 찾거나 설정하는 중 오류 발생. 기본 폰트를 사용합니다. 오류: {e}")
        # traceback.print_exc()
        plt.rcParams['axes.unicode_minus'] = False
        return None

def visualize_augmentation_stats(original_image_dir, augmented_image_dir, metadata_dir, output_visualization_dir, coco_json_path=None):
    """
    증강 통계를 계산하고 시각화합니다.
    """
    print("데이터 증강 통계 시각화를 시작합니다...")
    print(f"  원본 이미지 디렉토리 (참고용): {original_image_dir}")
    print(f"  증강 이미지 디렉토리: {augmented_image_dir}")
    print(f"  메타데이터 디렉토리: {metadata_dir}")
    print(f"  시각화 출력 디렉토리: {output_visualization_dir}")

    os.makedirs(output_visualization_dir, exist_ok=True)
    korean_font_prop = get_korean_font() # 한글 폰트 설정 시도

    # class_specific_stats_data 초기화 및 cat_id_to_name 초기화
    class_specific_stats_data = None
    cat_id_to_name = {} # 카테고리 ID와 이름 매핑

    # COCO JSON 및 stats.json 로드 (클래스별 통계에 필요)
    if coco_json_path:
        stats_json_path = os.path.join(metadata_dir, "stats.json")
        if not os.path.exists(coco_json_path):
            print(f"경고: COCO JSON 파일 \'{coco_json_path}\'를 찾을 수 없습니다. 클래스별 통계는 생성되지 않습니다.")
        elif not os.path.exists(stats_json_path):
            print(f"경고: 통계 파일 \'{stats_json_path}\'를 찾을 수 없습니다. 클래스별 통계는 생성되지 않습니다.")
        else:
            try:
                with open(coco_json_path, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                if 'categories' in coco_data and isinstance(coco_data['categories'], list):
                    # str(category['id'])를 키로 사용 (stats.json의 클래스 ID가 문자열이므로 일관성 유지)
                    cat_id_to_name = {str(category['id']): category['name'] for category in coco_data['categories']}
                    if not cat_id_to_name:
                         print(f"경고: COCO JSON 파일 \'{coco_json_path}\'에 카테고리 정보가 없거나 비어있습니다.")
                else:
                    print(f"경고: COCO JSON 파일 \'{coco_json_path}\'에 'categories' 키가 없거나 유효한 리스트가 아닙니다.")

                with open(stats_json_path, 'r', encoding='utf-8') as f:
                    class_specific_stats_data = json.load(f) # stats.json 내용을 로드
                
                if not class_specific_stats_data:
                    print(f"정보: 통계 파일 \'{stats_json_path}\'가 비어있습니다.")
                elif not cat_id_to_name: # 카테고리 정보가 없으면 클래스별 통계 의미 없음
                    print("경고: COCO JSON에서 클래스 이름을 가져올 수 없어 클래스별 통계를 정확히 생성할 수 없습니다.")
                    class_specific_stats_data = None # 클래스 이름 매핑 불가 시 통계 사용 안 함

            except json.JSONDecodeError as e:
                print(f"오류: JSON 파일 파싱 실패 (\'{coco_json_path}\' 또는 \'{stats_json_path}\') - {e}")
                class_specific_stats_data = None # 오류 발생 시 None으로 유지
            except Exception as e:
                print(f"오류: COCO JSON 또는 stats.json 파일 로드 중 예외 발생 - {e}")
                traceback.print_exc()
                class_specific_stats_data = None # 오류 발생 시 None으로 유지
    else:
        print("정보: --coco_json_path가 제공되지 않아 클래스별 통계는 생성되지 않습니다.")

    # 1. 메타데이터 로드 (mapping.csv)
    mapping_file_path = os.path.join(metadata_dir, "mapping.csv")
    if not os.path.exists(mapping_file_path):
        print(f"오류: 메타데이터 파일 '{mapping_file_path}'를 찾을 수 없습니다.")
        return

    try:
        df_mapping = pd.read_csv(mapping_file_path)
        if not all(col in df_mapping.columns for col in ['original_path', 'augmented_path', 'augmentation_type']):
            print(f"오류: '{mapping_file_path}' 파일에 필요한 컬럼(original_path, augmented_path, augmentation_type)이 없습니다.")
            return
    except Exception as e:
        print(f"오류: 메타데이터 파일 로드 실패 - {e}")
        traceback.print_exc()
        return

    # 2. 통계 계산
    num_augmented_images = len(df_mapping)
    unique_original_images = df_mapping['original_path'].unique()
    num_original_images_in_map = len(unique_original_images) # 매핑 파일에 기록된 원본 수

    augmentation_type_counts = Counter(df_mapping['augmentation_type'])

    print(f"\n--- 기본 통계 ---")
    print(f"매핑 파일 기준 고유 원본 이미지 수: {num_original_images_in_map}")
    print(f"총 생성된 증강 이미지 수: {num_augmented_images}")
    if num_original_images_in_map > 0 and num_augmented_images > 0:
        print(f"원본 1개당 평균 증강 이미지 수: {num_augmented_images / num_original_images_in_map:.2f}")
    else:
        print("증강 이미지가 없거나 매핑된 원본 이미지가 없어 평균을 계산할 수 없습니다.")

    # 3. 시각화
    plot_font_prop_dict = {} # title, xlabel, ylabel 등에 사용될 폰트 속성 딕셔너리
    textprops_for_pie = None # pie 차트의 textprops에 사용될 딕셔너리
    if korean_font_prop:
        plot_font_prop_dict = {'fontproperties': korean_font_prop}
        textprops_for_pie = {'fontproperties': korean_font_prop} # pie를 위한 textprops

    # 3.1. 파이 차트: 원본 vs 증강 비율 (매핑 파일 기준 원본 수 사용)
    if num_original_images_in_map > 0 or num_augmented_images > 0:
        labels_total = ['Original Images (in map)', 'Augmented Images'] 
        sizes_total = [num_original_images_in_map, num_augmented_images]
        explode_total = (0, 0.1) if num_original_images_in_map > 0 and num_augmented_images > 0 else (0,0)

        fig1, ax1 = plt.subplots(figsize=(10, 7))
        wedges, texts, autotexts = ax1.pie(sizes_total, explode=explode_total, labels=labels_total, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops=textprops_for_pie)
        ax1.axis('equal') 
        plt.title('Dataset Composition (Original vs. Augmented)', **plot_font_prop_dict)
        pie_chart_path1 = os.path.join(output_visualization_dir, "pie_chart_total_composition.png")
        try:
            plt.savefig(pie_chart_path1)
            print(f"  저장됨: {pie_chart_path1}")
        except Exception as e:
            print(f"오류: 파이 차트(전체 구성) 저장 실패 - {e}")
            traceback.print_exc()
        plt.close(fig1)

    # 3.2. 파이 차트: 증강 유형별 비율
    if augmentation_type_counts:
        labels_aug_types = list(augmentation_type_counts.keys())
        sizes_aug_types = list(augmentation_type_counts.values())
        
        fig2, ax2 = plt.subplots(figsize=(12, 8)) 
        wedges, texts, autotexts = ax2.pie(sizes_aug_types, autopct='%1.1f%%',
                                           shadow=True, startangle=90, 
                                           labels=labels_aug_types if not korean_font_prop else None, 
                                           textprops=textprops_for_pie)
        ax2.axis('equal')
        plt.title('Distribution of Augmented Images by Type', **plot_font_prop_dict)
        
        if korean_font_prop:
            # 범례의 title에도 폰트 적용을 위해 fontproperties 직접 명시
            ax2.legend(wedges, labels_aug_types,
                      title="Augmentation Type",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      prop=korean_font_prop, title_fontproperties=korean_font_prop)
        else: 
            pass

        plt.tight_layout()
        pie_chart_path2 = os.path.join(output_visualization_dir, "pie_chart_augmentation_types.png")
        try:
            plt.savefig(pie_chart_path2)
            print(f"  저장됨: {pie_chart_path2}")
        except Exception as e:
            print(f"오류: 파이 차트(유형별) 저장 실패 - {e}")
            traceback.print_exc()
        plt.close(fig2)

    # 3.3. 막대 그래프: 증강 유형별 생성된 이미지 수
    if augmentation_type_counts:
        sorted_aug_items = sorted(augmentation_type_counts.items(), key=lambda item: item[1], reverse=True)
        types = [item[0] for item in sorted_aug_items]
        counts = [item[1] for item in sorted_aug_items]

        fig3, ax3 = plt.subplots(figsize=(12, 8)) 
        bars = ax3.bar(types, counts, color='skyblue')
        ax3.set_xlabel('Augmentation Type', **plot_font_prop_dict)
        ax3.set_ylabel('Number of Generated Images', **plot_font_prop_dict)
        ax3.set_title('Number of Images Generated per Augmentation Type', **plot_font_prop_dict)
        plt.xticks(rotation=45, ha="right", **plot_font_prop_dict) 

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(counts) if counts else yval + 1 , int(yval), ha='center', va='bottom', **plot_font_prop_dict)

        plt.tight_layout()
        bar_chart_path = os.path.join(output_visualization_dir, "bar_chart_augmentation_counts.png")
        try:
            plt.savefig(bar_chart_path)
            print(f"  저장됨: {bar_chart_path}")
        except Exception as e:
            print(f"오류: 막대 그래프 저장 실패 - {e}")
            traceback.print_exc()
        plt.close(fig3)
        
    # --- 새로운 클래스별 시각화 로직 ---
    if class_specific_stats_data:
        print("\n--- 클래스별 증강 통계 시각화 ---")

        # 데이터 변환: stats.json 형식을 DataFrame으로 (예상 형식: {'aug_type': {'class_id': count}})
        # DataFrame 만들기: rows=클래스, columns=증강 타입, values=카운트
        processed_stats = []
        all_aug_types_from_stats = set()
        all_class_ids_from_stats = set()

        for aug_type, class_counts in class_specific_stats_data.items():
            all_aug_types_from_stats.add(aug_type)
            for class_id_str, count in class_counts.items():
                all_class_ids_from_stats.add(class_id_str)
                class_name = cat_id_to_name.get(class_id_str, f"ID:{class_id_str}") # COCO 매핑 사용, 없으면 ID 사용
                processed_stats.append({'class_name': class_name, 'augmentation_type': aug_type, 'count': count})
        
        if not processed_stats:
            print("정보: stats.json에서 처리할 수 있는 클래스별 증강 데이터가 없습니다.")
        else:
            df_class_stats = pd.DataFrame(processed_stats)
            
            # 그래프 1: 클래스별 누적 막대 그래프
            try:
                pivot_df_class_stacked = df_class_stats.pivot_table(index='class_name', columns='augmentation_type', values='count', fill_value=0)
                if not pivot_df_class_stacked.empty:
                    fig4, ax4 = plt.subplots(figsize=(max(10, len(all_class_ids_from_stats) * 0.8), 8)) # 클래스 수에 따라 너비 조절
                    pivot_df_class_stacked.plot(kind='bar', stacked=True, ax=ax4, colormap='viridis')
                    
                    ax4.set_xlabel('Class', **plot_font_prop_dict)
                    ax4.set_ylabel('Number of Augmentations', **plot_font_prop_dict)
                    ax4.set_title('Augmentations per Class (Stacked)', **plot_font_prop_dict)
                    plt.xticks(rotation=45, ha="right", **plot_font_prop_dict)
                    ax4.legend(title='Augmentation Type', prop=korean_font_prop, title_fontproperties=korean_font_prop if korean_font_prop else None, bbox_to_anchor=(1.02, 1), loc='upper left')
                    
                    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 범례 공간 확보
                    stacked_bar_path = os.path.join(output_visualization_dir, "bar_chart_augmentations_per_class_stacked.png")
                    plt.savefig(stacked_bar_path)
                    plt.close(fig4)
                    print(f"  저장됨: {stacked_bar_path}")
                else:
                    print("정보: 클래스별 누적 막대 그래프를 생성할 데이터가 없습니다 (피봇 테이블 비어 있음).")
            except Exception as e:
                print(f"오류: 클래스별 누적 막대 그래프 생성/저장 실패 - {e}")
                traceback.print_exc()

            # 그래프 2: 증강 유형별 클래스 분포 막대 그래프
            try:
                pivot_df_aug_type_grouped = df_class_stats.pivot_table(index='augmentation_type', columns='class_name', values='count', fill_value=0)
                if not pivot_df_aug_type_grouped.empty:
                    fig5, ax5 = plt.subplots(figsize=(max(10, len(all_aug_types_from_stats) * 1.2), 8)) # 증강 유형 수에 따라 너비 조절
                    pivot_df_aug_type_grouped.plot(kind='bar', ax=ax5, colormap='Spectral') # stacked=False (기본값)
                    
                    ax5.set_xlabel('Augmentation Type', **plot_font_prop_dict)
                    ax5.set_ylabel('Number of Times Applied', **plot_font_prop_dict)
                    ax5.set_title('Class Distribution per Augmentation Type', **plot_font_prop_dict)
                    plt.xticks(rotation=45, ha="right", **plot_font_prop_dict)
                    ax5.legend(title='Class', prop=korean_font_prop, title_fontproperties=korean_font_prop if korean_font_prop else None, bbox_to_anchor=(1.02, 1), loc='upper left')
                    
                    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 범례 공간 확보
                    grouped_bar_path = os.path.join(output_visualization_dir, "bar_chart_class_dist_per_aug_type.png")
                    plt.savefig(grouped_bar_path)
                    plt.close(fig5)
                    print(f"  저장됨: {grouped_bar_path}")
                else:
                    print("정보: 증강 유형별 클래스 분포 그래프를 생성할 데이터가 없습니다 (피봇 테이블 비어 있음).")
            except Exception as e:
                print(f"오류: 증강 유형별 클래스 분포 그래프 생성/저장 실패 - {e}")
                traceback.print_exc()

    print("\n시각화 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터 증강 결과 시각화 스크립트")
    parser.add_argument("--original_image_dir", type=str, required=False, help="(선택 사항) 원본 이미지가 저장된 디렉토리 경로. 현재 스크립트에서는 직접 사용되지 않으나, 참고용으로 받을 수 있습니다.")
    parser.add_argument("--augmented_image_dir", type=str, required=True, help="시각화 대상인 증강된 이미지들이 저장된 디렉토리 경로입니다.")
    parser.add_argument("--metadata_dir", type=str, required=True, help="증강 메타데이터(예: mapping.csv, stats.json)가 포함된 디렉토리 경로입니다.")
    parser.add_argument("--output_visualization_dir", type=str, required=True, help="생성된 시각화 결과(이미지 파일)를 저장할 디렉토리 경로입니다.")
    parser.add_argument("--coco_json_path", type=str, required=False, help="(선택 사항) 원본 COCO JSON 파일 경로. 클래스 ID와 이름을 매핑하는 데 사용됩니다.")
    
    args = parser.parse_args()

    visualize_augmentation_stats(
        args.original_image_dir,
        args.augmented_image_dir,
        args.metadata_dir,
        args.output_visualization_dir,
        args.coco_json_path
    ) 