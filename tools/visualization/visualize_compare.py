# tools/visualize_compare.py

from PIL import Image, ImageDraw, ImageFont

# TrueType 폰트 로드 (시스템에 있는 한글폰트 경로로 수정)
def _get_kr_font(size=14):
    # 예) 윈도우: C:\Windows\Fonts\malgun.ttf
    #     우분투  : /usr/share/fonts/truetype/nanum/NanumGothic.ttf
    for path in [
        'C:/Users/추동명/AppData/Local/Microsoft/Windows/Fonts/Pretendard-Medium.otf',
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",          # Ubuntu
        "C:/Windows/Fonts/malgun.ttf",                              # Windows
        "C:/Windows/Fonts/NanumGothic.ttf",
    ]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size, encoding="utf-8")
    raise FileNotFoundError("한글 TrueType 폰트를 찾을 수 없습니다. 경로를 수정하세요.")

import os, sys, argparse
from pathlib import Path

import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]         # 프로젝트 루트
sys.path.append(str(ROOT))

from models.yolo_v8 import YOLOModel
from src.data_preprocessing import apply_custom_preprocessing

def _get_text_size(draw, text, font):
    """
    Pillow 9.x   : draw.textsize()
    Pillow 10.x+ : draw.textbbox()
    """
    if hasattr(draw, "textbbox"):                 # Pillow ≥10
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    else:                                         # Pillow <10
        return draw.textsize(text, font=font)
    
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "경차/세단","SUV/승합차","트럭","버스(소형, 대형)","통학버스(소형,대형)",
    "경찰차","구급차","소방차","견인차","기타 특장차",
    "성인","어린이","오토바이","자전거 / 기타 전동 이동체","라바콘","삼각대","기타",
]
COLORS = [tuple(int(x*255) for x in cv2.cvtColor(
           np.uint8([[[(i*15)%180,255,200]]]), cv2.COLOR_HSV2BGR)[0,0]/255)
          for i in range(len(CLASS_NAMES))]

# ──────────────────────────────────────────────────────────────
def yolo_txt_to_boxes(txt_path, w, h):
    boxes=[]
    if not txt_path.is_file(): return boxes
    for ln in txt_path.read_text().splitlines():
        parts=ln.strip().split()
        if len(parts)<5: continue
        cid,xc,yc,bw,bh=map(float,parts[:5])
        x1=(xc-bw/2)*w; y1=(yc-bh/2)*h
        x2=(xc+bw/2)*w; y2=(yc+bh/2)*h
        boxes.append([x1,y1,x2,y2,int(cid)])
    return boxes

# ──────────────────────────────────────────────────────────────
def draw_boxes(img_bgr, boxes, with_score=False, font_scale=.5):
    """
    img_bgr : OpenCV BGR 이미지 (numpy.ndarray)
    boxes   : [x1,y1,x2,y2,(score,)cls_id] 형식
    반환     : 바운딩박스‧라벨이 그려진 BGR 이미지
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)       # → RGB
    img_pil = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(img_pil)
    font_sz = max(12, int(16*font_scale))
    font    = _get_kr_font(font_sz)

    for b in boxes:
        x1,y1,x2,y2 = map(int, b[:4])
        cid         = int(b[5] if with_score else b[4])
        score       = b[4]      if with_score else None
        label       = CLASS_NAMES[cid] + (f" {score:.2f}" if score is not None else "")
        color_rgb   = tuple(int(c*255) for c in COLORS[cid])      # PIL(RGB)용

        # ─ 바운딩 박스 & 레이블 ────────────────────────────────
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)

        tw, th = _get_text_size(draw, label, font)
        draw.rectangle([x1, y1-th-2, x1+tw+4, y1], fill=color_rgb)
        draw.text((x1+2, y1-th-2), label, font=font, fill=(0,0,0))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)     # ← 다시 BGR

# ──────────────────────────────────────────────────────────────
def main(a):
    device = "0" if torch.cuda.is_available() else "cpu"
    model1 = YOLOModel(model_name=a.weights1, device=device)
    model2 = YOLOModel(model_name=a.weights2, device=device)
    print("[INFO] two models loaded.")

    tdir     = Path(a.test_dir)
    img_dir  = tdir / ("images_processed" if (tdir/"images_processed").is_dir() else "images")
    gt_dir   = tdir / ("labels_yolo"      if (tdir/"labels_yolo").is_dir()      else "labels")
    out_dir  = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([*img_dir.glob("*.jpg"), *img_dir.glob("*.png"), *img_dir.glob("*.jpeg")])

    for idx, p in enumerate(imgs, 1):
        img = cv2.imread(str(p));  h, w = img.shape[:2]

        # ───────────────── GT ─────────────────
        gt_img = draw_boxes(img.copy(),
                            yolo_txt_to_boxes(gt_dir/(p.stem + ".txt"), w, h))

        # ────────────── Model-1 ───────────────
        pre     = apply_custom_preprocessing(img, use_normalization=False) if a.preprocess else img
        pred1   = model1.predict(pre, conf_thresh=a.conf1)
        pred1_img = draw_boxes(img.copy(), pred1, with_score=True)

        # ────────────── Model-2 ───────────────
        pred2   = model2.predict(pre, conf_thresh=a.conf2)
        pred2_img = draw_boxes(img.copy(), pred2, with_score=True)

        # ─── 중앙 상단에 제목 넣기 ──────────────
        titles = [("GT", (255,255,255)),
                  (f"Model-1  (thr={a.conf1})", (255,255,255)),
                  (f"Model-2  (thr={a.conf2})", (255,255,255))]

        for canvas, (title, color) in zip([gt_img, pred1_img, pred2_img], titles):
            cv2.putText(canvas,
                        title,
                        (canvas.shape[1]//2 - 110, 28),         # 대략 가운데
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        thickness=2,
                        lineType=cv2.LINE_AA)

        # ─── 3장 가로로 붙여 저장 ──────────────
        concat = np.concatenate([gt_img, pred1_img, pred2_img], axis=1)
        cv2.imwrite(str(out_dir / f"{p.stem}_compare.jpg"), concat)

        if idx % 100 == 0 or idx == len(imgs):
            print(f"[{idx}/{len(imgs)}] saved {p.stem}_compare.jpg")

    print(f"\nDone. Results → {out_dir}")

# ──────────────────────────────────────────────────────────────
if __name__=="__main__":
    pa=argparse.ArgumentParser(description="GT vs two YOLO models visual comparison")
    pa.add_argument("--test_dir", required=True, help="dataset folder with images/ & labels_yolo/")
    pa.add_argument("--weights1", required=True, help="model-1 .pt")
    pa.add_argument("--weights2", required=True, help="model-2 .pt")
    pa.add_argument("--conf1", type=float, default=0.15, help="confidence threshold for model-1")
    pa.add_argument("--conf2", type=float, default=0.15, help="confidence threshold for model-2")
    pa.add_argument("--preprocess", action="store_true", help="apply CLAHE + blur before inference")
    pa.add_argument("--out_dir", default="visual_compare", help="output folder")
    main(pa.parse_args())
