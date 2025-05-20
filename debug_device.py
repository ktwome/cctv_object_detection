import torch
from models.dfine_b import DFineModel

# 모델 생성
model = DFineModel(num_classes=17, device="cuda" if torch.cuda.is_available() else "cpu")

# 디바이스 정보 출력
print(f"모델 디바이스: {model.device}")
print(f"디바이스 타입: {type(model.device)}")
if isinstance(model.device, torch.device):
    print(f"디바이스 타입 속성: {model.device.type}")

# PyTorch 버전 확인
print(f"PyTorch 버전: {torch.__version__}") 