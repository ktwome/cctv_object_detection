import torch
import sys
import yaml
import os

def print_args_map(pt_path, args_yaml_path=None):
    print(f"== PT 파일: {pt_path}")
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    train_args = ckpt.get('train_args', {})
    print("\n[train_args in pt 파일]")
    for k, v in train_args.items():
        print(f"  {k}: {v}")

    if args_yaml_path and os.path.exists(args_yaml_path):
        print(f"\n== args.yaml: {args_yaml_path}")
        with open(args_yaml_path, encoding='utf-8') as f:
            args_yaml = yaml.safe_load(f)
        print("[args.yaml 내용]")
        for k, v in args_yaml.items():
            print(f"  {k}: {v}")
        # 차이점 출력
        print("\n[차이점]")
        all_keys = set(train_args.keys()) | set(args_yaml.keys())
        for k in sorted(all_keys):
            pt_v = train_args.get(k, '<없음>')
            yaml_v = args_yaml.get(k, '<없음>')
            if pt_v != yaml_v:
                print(f"  {k}: pt={pt_v} | yaml={yaml_v}")
    else:
        print("\n(args.yaml 파일이 없거나 경로 미지정)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python pt_args_map.py <pt파일경로> [args.yaml 경로]")
    else:
        pt_path = sys.argv[1]
        args_yaml_path = sys.argv[2] if len(sys.argv) > 2 else None
        print_args_map(pt_path, args_yaml_path) 