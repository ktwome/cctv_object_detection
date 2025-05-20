import torch

print('CUDA 사용 가능:', torch.cuda.is_available())
print('CUDA 버전:', torch.version.cuda)
print('PyTorch 버전:', torch.__version__)
print('GPU 개수:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('현재 GPU:', torch.cuda.current_device())
    print('GPU 이름:', torch.cuda.get_device_name(0))
else:
    print('현재 GPU: None')
    print('GPU 이름: None')