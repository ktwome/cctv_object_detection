import torch
import sys

def sync_train_args_to_args(pt_path, save_path=None):
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    train_args = ckpt.get('train_args', {})
    # args 네임스페이스가 없으면 새로 생성
    if 'args' not in ckpt or not isinstance(ckpt['args'], dict):
        ckpt['args'] = {}
    # train_args의 내용을 args에 복사
    for k, v in train_args.items():
        ckpt['args'][k] = v
    print(f"[INFO] train_args의 {len(train_args)}개 key를 args 네임스페이스에 복사 완료")
    # 저장
    if save_path:
        torch.save(ckpt, save_path)
        print(f"[INFO] 수정된 pt 파일을 {save_path}에 저장 완료")
    else:
        torch.save(ckpt, pt_path)
        print(f"[INFO] 원본 pt 파일({pt_path})을 덮어씀")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python pt_sync_args.py <pt파일경로> [저장경로]")
    else:
        pt_path = sys.argv[1]
        save_path = sys.argv[2] if len(sys.argv) > 2 else None
        sync_train_args_to_args(pt_path, save_path)