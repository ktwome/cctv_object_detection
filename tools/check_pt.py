import torch
import sys

def print_pt_info(pt_path):
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    print("== PT 파일 주요 정보 ==")
    for k in ['epoch', 'best_fitness', 'train_args', 'model', 'optimizer']:
        if k in ckpt:
            print(f"{k}: {ckpt[k] if k != 'model' and k != 'optimizer' else '...생략...' }")
    print("== 전체 키 목록 ==")
    print(list(ckpt.keys()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python check_pt.py <pt파일경로>")
    else:
        print_pt_info(sys.argv[1]) 