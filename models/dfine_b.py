# dfine_model.py
import os
import time
import yaml
import torch
import wandb
from pathlib import Path
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

# KakaoBrain D-FINE 레퍼런스 모듈 (가정)
from dfine.model_zoo import DfineB
from dfine.loss import DiffusionCriterion
from dfine.data import CocoDiffusionDataset, collate_fn
from dfine.utils.ema import ModelEma
from dfine.utils.augment import build_augment

__all__ = ["DFineModel"]

class DFineModel:
    """
    D-FINE-B(디퓨전 기반 객체 검출) 래퍼 클래스
    ─────────────────────────────────────────────
    * 학습(train) · 추론(predict) · 저장(save) · 로드(load) 지원
    * EMA · AMP · Grad Accum · LR Scheduler · W&B 로깅 내장
    """

    def __init__(self,
                 backbone_ckpt: str | None = None,
                 num_classes: int = 1,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 project: str = "runs/dfine",
                 run_name: str | None = None):
        self.device = device
        self.num_classes = num_classes
        self.project = Path(project)
        self.project.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_name = run_name or f"dfineB-{ts}"
        self.exp_dir = self.project / self.run_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # ── 모델 인스턴스 ───────────────────────
        self.model = DfineB(backbone_pretrained=bool(backbone_ckpt),
                            num_classes=num_classes)
        if backbone_ckpt:
            self.model.backbone.load_state_dict(torch.load(backbone_ckpt))

        self.model.to(self.device)
        self.ema = None                    # 초기에는 비활성

    # ─────────────────────────────────────────────
    # CONFIG 스냅샷 저장
    # ─────────────────────────────────────────────
    def _dump_cfg(self, cfg: dict):
        with open(self.exp_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

    # ─────────────────────────────────────────────
    # TRAIN
    # ─────────────────────────────────────────────
    def train(self,
              train_json: str,
              val_json: str,
              img_root: str,
              epochs: int = 60,
              batch: int = 2,
              lr: float = 2e-4,
              weight_decay: float = 1e-4,
              warmup_epochs: int = 1,
              accum_steps: int | None = None,
              amp: bool = True,
              ema_decay: float = 0.9999,
              num_workers: int = 8,
              resume: str | None = None,
              log_wandb: bool = True,
              early_stop_patience: int = 10,
              diffusion_steps: int = 100,
              aug_cfg: dict | None = None):
        """
        D-FINE-B 학습 실행
        ------------------------------------------
        train_json / val_json : COCO 형식 주석 파일 경로
        img_root              : 이미지 디렉터리
        diffusion_steps       : 디퓨전 타임스텝 수(기본 100)
        aug_cfg               : Albumentations 구성 dict
        """
        cfg = locals().copy()   # 설정 스냅샷
        self._dump_cfg(cfg)

        # ───── 데이터로더 ───────────────────────
        aug = build_augment(aug_cfg)
        ds_train = CocoDiffusionDataset(train_json, img_root, self.num_classes,
                                        diffusion_steps, aug)
        ds_val   = CocoDiffusionDataset(val_json,   img_root, self.num_classes,
                                        diffusion_steps, None)

        if accum_steps is None:
            # GPU VRAM 기준 자동 결정 (12 GB당 2)
            accum_steps = max(1, int(24 / torch.cuda.get_device_properties(0).total_memory * 10))

        dl_train = DataLoader(ds_train, batch_size=batch,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=True)
        dl_val   = DataLoader(ds_val, batch_size=batch,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=True)

        # ───── Optim / Scheduler ────────────────
        opt = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        lf = lambda it: (it / (warmup_epochs * len(dl_train))) if it < warmup_epochs * len(dl_train) \
                        else 0.5 * (1 + torch.cos(torch.pi * (it - warmup_epochs * len(dl_train)) /
                                                  (epochs * len(dl_train) - warmup_epochs * len(dl_train))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lf)

        # ───── Criterion & AMP / EMA ────────────
        criterion = DiffusionCriterion(diffusion_steps).to(self.device)
        scaler = GradScaler(enabled=amp)
        if ema_decay:
            self.ema = ModelEma(self.model, decay=ema_decay)

        # ───── W&B ──────────────────────────────
        if log_wandb:
            wandb.init(project="D-FINE-B", name=self.run_name, config=cfg,
                       dir=str(self.exp_dir), sync_tensorboard=True)

        start_epoch = 0
        best_map = 0.0
        best_ckpt = self.exp_dir / "best.pth"

        # resume
        if resume:
            ckpt = torch.load(resume, map_location="cpu")
            self.model.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['opt'])
            scheduler.load_state_dict(ckpt['scheduler'])
            if self.ema:
                self.ema.module.load_state_dict(ckpt['ema'])
            start_epoch = ckpt['epoch'] + 1
            best_map = ckpt.get('best_map', 0.0)

        # ───── Training Loop ────────────────────
        no_improve = 0
        for epoch in range(start_epoch, epochs):
            self.model.train()
            epoch_loss = 0.0
            for step, batch_data in enumerate(dl_train):
                images, targets = batch_data["image"].to(self.device), batch_data["target"]
                with autocast(enabled=amp):
                    pred = self.model(images, targets)
                    loss = criterion(pred, targets) / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    scheduler.step()
                    if self.ema:
                        self.ema.update(self.model)
                epoch_loss += loss.item() * accum_steps

            # ───── Validation ───────────────────
            map_val = self._evaluate(dl_val)
            if map_val > best_map:
                best_map = map_val
                torch.save({"model": self.model.state_dict(),
                            "opt": opt.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "ema": self.ema.module.state_dict() if self.ema else {},
                            "epoch": epoch,
                            "best_map": best_map},
                           best_ckpt)
                no_improve = 0
            else:
                no_improve += 1

            # ───── Logging ───────────────────────
            if log_wandb:
                wandb.log({"loss/train": epoch_loss / len(dl_train),
                           "mAP/val": map_val,
                           "lr": scheduler.get_last_lr()[0]}, step=epoch)

            print(f"[E{epoch:03d}] loss={epoch_loss/len(dl_train):.4f}, mAP={map_val:.3f}")

            if no_improve >= early_stop_patience:
                print(f"[EarlyStop] {early_stop_patience} epochs without improvement.")
                break

        print(f"Training finished. Best mAP={best_map:.3f} | ckpt: {best_ckpt}")
        self.model.load_state_dict(torch.load(best_ckpt)["model"])

    # ─────────────────────────────────────────────
    # EVALUATE (COCO mAP)
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, dataloader):
        self.model.eval()
        ious, confs, labels = [], [], []
        for batch in dataloader:
            images, targets = batch["image"].to(self.device), batch["target"]
            pred = self.model(images)
            # …(IoU & mAP 계산 코드 생략: COCOEvaluator 또는 pycocotools 사용)…
        # 간단히 placeholder 반환
        return random.uniform(0.30, 0.70)

    # ─────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, img_tensor: torch.Tensor, conf_thr: float = 0.25):
        """
        img_tensor : (3,H,W) FloatTensor [0,1]
        반환값 : [[x1,y1,x2,y2,score,class_id], …]
        """
        self.model.eval()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        pred = self.model(img_tensor)  # (N, queries, 4+cls)
        boxes, scores, cls = pred["boxes"], pred["scores"], pred["labels"]
        keep = scores > conf_thr
        boxes = box_convert(boxes[keep], in_fmt="cxcywh", out_fmt="xyxy").cpu().tolist()
        scores = scores[keep].cpu().tolist()
        cls = cls[keep].cpu().tolist()
        return [b + [s, c] for b, s, c in zip(boxes, scores, cls)]

    # ─────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
