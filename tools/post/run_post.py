import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2])) 

from tools.post.post_processing import sweep_thresholds_and_evaluate

metrics = sweep_thresholds_and_evaluate(
    model_weights='src/jsjang.pt',
    test_dir='aug_real_pro_dataset/test_aug',          # images / labels_yolo 포함 폴더
    thresholds=[0.25],   # 필요 시 커스터마이즈
    device='cuda',                             # 'auto'·'cpu' 도 가능
    preprocess=True,                           # on-the-fly 전처리 사용 여부
    keep_best_pred_dir='best_predictions',     # 생략 가능
    debug=True                                 # 상세 로그 출력
)

print(metrics)