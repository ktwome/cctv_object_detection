"""
D-FINE-B 모델 래퍼 클래스

이 모듈은 D-FINE-B 모델을 다른 코드와 통합하기 위한 래퍼 클래스를 제공합니다.
"""

import os
import time
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# 선택적 wandb 가져오기
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[경고] wandb 모듈을 찾을 수 없습니다. 로깅이 비활성화됩니다.")

# 내부 구현 모듈 가져오기
from src.d_fine import (
    DFineB, 
    CocoDiffusionDataset, 
    collate_fn, 
    DiffusionCriterion, 
    ModelEma, 
    build_augment
)

__all__ = ["DFineModel"]


class DFineModel:
    """
    D-FINE-B(디퓨전 기반 객체 검출) 래퍼 클래스
    
    기능:
    - 학습(train): COCO 형식 데이터셋으로 모델 학습
    - 추론(predict): 이미지에서 객체 감지
    - 저장(save): 모델 가중치 저장
    - 로드(load): 모델 가중치 로드
    
    추가 기능:
    - EMA: 모델 가중치의 지수이동평균
    - AMP: 자동 혼합 정밀도 학습
    - Grad Accum: 그래디언트 누적
    - LR Scheduler: 학습률 스케줄러
    - W&B 로깅: (선택적) Weights & Biases 로깅
    """

    def __init__(self,
                 backbone_ckpt=None,
                 num_classes=1,
                 device=None,
                 project="runs/dfine",
                 run_name=None):
        """
        D-FINE-B 모델 초기화
        
        Args:
            backbone_ckpt: 백본 체크포인트 경로 (optional)
            num_classes: 클래스 수
            device: 사용할 디바이스 ('cuda' 또는 'cpu')
            project: 결과 저장 디렉토리
            run_name: 실행 이름 (None이면 자동 생성)
        """
        # 디바이스 설정
        if device is None:
            # 사용 가능한 GPU 확인
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # 문자열이면 torch.device로 변환
            self.device = device if isinstance(device, torch.device) else torch.device(device)
            
        print(f"모델 디바이스: {self.device}")
        
        self.num_classes = num_classes
        self.project = Path(project)
        self.project.mkdir(parents=True, exist_ok=True)

        # 실행 이름 및 디렉토리 설정
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_name = run_name or f"dfineB-{ts}"
        self.exp_dir = self.project / self.run_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 모델 인스턴스 생성
        self.model = DFineB(backbone_pretrained=backbone_ckpt is not None,
                            num_classes=num_classes)
                           
        # 백본 체크포인트 로드 (있는 경우)
        if backbone_ckpt and os.path.exists(backbone_ckpt):
            print(f"백본 체크포인트를 로드합니다: {backbone_ckpt}")
            # 백본 부분만 로드
            self.model.backbone.load_state_dict(torch.load(backbone_ckpt, map_location=self.device))

        # 모델을 지정된 디바이스로 이동
        self.model.to(self.device)
        self.ema = None  # EMA는 학습 시 초기화됨

    def _dump_cfg(self, cfg):
        """설정 스냅샷 저장"""
        # 복잡한 객체는 문자열로 변환
        simplified_cfg = {}
        for k, v in cfg.items():
            if k in ['self', 'targets', 'aug_cfg'] or k.startswith('__'):
                continue
            try:
                yaml.safe_dump({k: v})
                simplified_cfg[k] = v
            except (TypeError, yaml.representer.RepresenterError):
                simplified_cfg[k] = str(v)
                
        # YAML 파일로 저장
        with open(self.exp_dir / "config.yaml", "w", encoding='utf-8') as f:
            yaml.safe_dump(simplified_cfg, f, allow_unicode=True)

    def train(self,
              train_json,
              val_json,
              img_root,
              epochs=60,
              batch=2,
              lr=2e-4,
              weight_decay=1e-4,
              warmup_epochs=1,
              accum_steps=None,
              amp=True,
              ema_decay=0.9999,
              num_workers=None,
              resume=None,
              log_wandb=False,
              early_stop_patience=10,
              diffusion_steps=100,
              aug_cfg=None,
              resize_to=(640, 640),
              pin_memory=True,
              prefetch_factor=2,
              monitor_memory=True,
              max_cache_size=1000,
              use_subset=False,
              subset_ratio=0.1,
              balanced_sampling=False):
        """
        D-FINE-B 모델 학습
        
        Args:
            train_json: COCO 형식 학습 주석 파일 경로
            val_json: COCO 형식 검증 주석 파일 경로
            img_root: 이미지 디렉토리 경로
            epochs: 학습 에포크 수
            batch: 배치 크기
            lr: 학습률
            weight_decay: 가중치 감쇠
            warmup_epochs: 워밍업 에포크 수
            accum_steps: 그래디언트 누적 스텝 수 (None이면 자동 결정)
            amp: 자동 혼합 정밀도 사용 여부
            ema_decay: EMA 감쇠율
            num_workers: 데이터 로더 워커 수 (None이면 자동 계산)
            resume: 학습 재개를 위한 체크포인트 경로
            log_wandb: W&B 로깅 사용 여부
            early_stop_patience: 조기 종료 인내심
            diffusion_steps: 디퓨전 타임스텝 수
            aug_cfg: 데이터 증강 설정 딕셔너리
            resize_to: 이미지 리사이즈 크기 (모든 이미지를 동일한 크기로 처리)
            pin_memory: 메모리 고정 사용 여부
            prefetch_factor: 데이터 미리 가져오기 계수
            monitor_memory: GPU 메모리 사용량 모니터링 여부
            max_cache_size: 이미지 캐시 최대 크기 (MB)
            use_subset: 데이터셋의 일부만 사용할지 여부
            subset_ratio: 사용할 데이터셋 비율 (0.1=10%)
            balanced_sampling: 클래스 균형을 위한 가중치 샘플링 사용 여부
        """
        # 멀티프로세싱 설정 최적화
        import torch.multiprocessing as mp
        mp.set_sharing_strategy('file_system')  # 파일 시스템 기반 공유로 변경
        
        # 설정 스냅샷 저장
        cfg = locals().copy()
        self._dump_cfg(cfg)

        # 데이터 로더 워커 수 자동 계산
        if num_workers is None:
            if hasattr(os, 'sched_getaffinity'):  # Linux
                num_workers = len(os.sched_getaffinity(0))
            else:  # Windows, macOS
                num_workers = os.cpu_count()
                
            # 일반적으로 CPU 코어 수의 1/4이 최적 (최대 4개로 제한)
            num_workers = min(4, max(1, num_workers // 4))
        
        print(f"학습 설정:\n - COCO 주석 파일: {train_json}\n - 이미지 루트: {img_root}")
        print(f" - 에포크: {epochs}, 배치: {batch}, 학습률: {lr}")
        print(f" - AMP: {amp}, EMA: {ema_decay is not None}, W&B: {log_wandb}")
        print(f" - 이미지 크기: {resize_to}, 데이터 로더 워커: {num_workers}개")
        print(f" - 메모리 최적화: 캐시 크기 {max_cache_size}MB")
        
        if use_subset:
            print(f" - 서브셋 사용: {subset_ratio:.1%} 데이터 ({int(169148*subset_ratio)}개 이미지)")
        
        if balanced_sampling:
            print(f" - 균등 샘플링: 활성화 (클래스 불균형 완화)")

        # 데이터 증강 설정
        if aug_cfg is None:
            aug_cfg = {}
        
        # 크기 고정을 위한 설정 추가
        aug_cfg['fixed_size'] = resize_to
        
        aug_fn = build_augment(aug_cfg)
        
        # 데이터셋 및 로더 생성
        try:
            # 학습 데이터셋 로드
            ds_train = CocoDiffusionDataset(
                train_json, img_root, self.num_classes, diffusion_steps, aug_fn,
                cache_images=True, max_cache_size=max_cache_size, memory_efficient=True)
            
            # 검증 데이터셋 로드
            ds_val = CocoDiffusionDataset(
                val_json, img_root, self.num_classes, diffusion_steps, None,
                cache_images=True, max_cache_size=max_cache_size//2, memory_efficient=True)
            
            # 데이터 서브셋 적용 (개발/디버깅용)
            if use_subset:
                from torch.utils.data import Subset
                import random
                
                # 원본 데이터셋 크기
                full_size = len(ds_train)
                subset_size = int(full_size * subset_ratio)
                
                # 이미지 인덱스 샘플링
                indices = list(range(full_size))
                random.shuffle(indices)  # 랜덤 셔플
                subset_indices = indices[:subset_size]
                
                # 서브셋 생성
                ds_train = Subset(ds_train, subset_indices)
                print(f"서브셋 적용: {full_size} → {subset_size}개 이미지")
                
        except Exception as e:
            raise RuntimeError(f"데이터셋 로드 중 오류 발생: {e}")
            
        # 균등 샘플링 적용 (클래스 불균형 완화)
        sampler = None
        if balanced_sampling:
            from torch.utils.data import WeightedRandomSampler
            
            # 원본 데이터셋 가져오기 (Subset인 경우)
            dataset = ds_train.dataset if hasattr(ds_train, 'dataset') else ds_train
            
            # 클래스별 가중치 계산
            class_counts = dataset.get_class_counts()
            weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0)  # NaN, inf 방지
            
            # 타겟 가져오기
            if hasattr(ds_train, 'dataset'):  # Subset인 경우
                targets = dataset.get_all_targets()
                # 서브셋 인덱스에 맞춰 필터링
                subset_indices = ds_train.indices
                targets = [targets[i] for i in subset_indices]
            else:
                targets = dataset.get_all_targets()
            
            # 샘플 가중치 생성
            sample_weights = [weights[label].item() for label in targets]
            
            # 샘플러 생성
            sampler = WeightedRandomSampler(
                sample_weights, 
                len(sample_weights), 
                replacement=True
            )
            
            print(f"균등 샘플링 적용: {len(class_counts)}개 클래스 간 밸런싱")
        
        # 학습 최적화 도구
        if accum_steps is None:
            # GPU 메모리 기반 자동 그래디언트 누적 계산
            if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                gpu_vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)  # GB
                print(f"GPU VRAM: {gpu_vram:.1f} GB", end="")
                
                # VRAM 기반 누적 스텝 계산
                if gpu_vram < 6:
                    accum_steps = 4  # 6GB 미만
                elif gpu_vram < 12:
                    accum_steps = 2  # 6-12GB
                else:
                    accum_steps = 1  # 12GB 이상
                
                print(f", 자동 그래디언트 누적 스텝: {accum_steps}")
            else:
                accum_steps = 1
                print("GPU 없음, 그래디언트 누적 없음")
        
        # 데이터 로더 생성
        print(f"데이터 로더 생성 중 (배치 크기: {batch}, 워커 수: {num_workers})")
        dl_train = DataLoader(
            ds_train, 
            batch_size=batch, 
            shuffle=(sampler is None),  # 샘플러 사용 시 셔플 비활성화
            sampler=sampler,  # 균등 샘플링 샘플러 (없으면 None)
            num_workers=num_workers, 
            collate_fn=collate_fn, 
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,  # 워커를 유지하여 재생성 오버헤드 감소
            drop_last=True)
        
        dl_val = DataLoader(
            ds_val, batch_size=batch, shuffle=False, 
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0)  # 워커를 유지

        # GPU 메모리 최적화
        if self.device.type == 'cuda':
            # 메모리 캐시 비우기
            torch.cuda.empty_cache()
            
            # 메모리 할당자 설정 최적화
            if hasattr(torch.cuda, 'memory_stats'):
                print("CUDA 메모리 최적화 설정 적용")
                # 메모리 조각화 감소
                torch.cuda.set_per_process_memory_fraction(0.9)  # 최대 90%만 사용
                
                # 초기 메모리 통계
                mem_stats = torch.cuda.memory_stats()
                reserved_mb = mem_stats.get('reserved_bytes.all.current', 0) / (1024**2)
                allocated_mb = mem_stats.get('allocated_bytes.all.current', 0) / (1024**2)
                print(f"GPU 초기 메모리: {allocated_mb:.0f}MB (할당) / {reserved_mb:.0f}MB (예약)")
        
        # 총 학습 스텝 수 계산
        total_batches = len(dl_train)
        total_steps = epochs * total_batches
        print(f"총 학습 스텝: {total_steps}회 (에포크당 {total_batches}회)")
        
        # 손실 함수 및 옵티마이저 설정
        criterion = DiffusionCriterion(diffusion_steps=diffusion_steps)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 학습률 스케줄러
        total_steps = epochs * len(dl_train) // accum_steps
        warmup_steps = warmup_epochs * len(dl_train) // accum_steps
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Automatic Mixed Precision 설정
        # 이전 코드: scaler = torch.cuda.amp.GradScaler(enabled=amp)
        # 호환성을 위한 코드
        scaler = torch.amp.GradScaler(enabled=amp)
        
        # EMA 설정
        if ema_decay > 0:
            print(f"EMA 모델 사용 (decay={ema_decay})")
            self.ema = ModelEma(self.model, decay=ema_decay, device=self.device)
        else:
            self.ema = None
            
        # 체크포인트 경로
        out_dir = self.project / self.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = out_dir / "best.pth"
        last_ckpt = out_dir / "last.pth"
        
        # 학습 환경 저장
        self._dump_cfg({
            "epochs": epochs,
            "batch_size": batch,
            "lr": lr,
            "weight_decay": weight_decay,
            "amp": amp,
            "ema_decay": ema_decay,
            "diffusion_steps": diffusion_steps,
            "resize_to": resize_to,
        })
        
        # W&B 초기화
        if log_wandb and WANDB_AVAILABLE:
            run_id = Path(resume).parent.name if resume else None
            
            wandb.init(
                project="d-fine",
                name=self.run_name,
                config={
                    "epochs": epochs,
                    "batch_size": batch,
                    "lr": lr,
                    "model": "dfine-b",
                },
                id=run_id,
                resume="allow" if run_id else None
            )
        
        # 체크포인트 로드 (있는 경우)
        start_epoch = 0
        best_map = 0.0

        if resume:
            if Path(resume).exists():
                print(f"체크포인트 로드 중: {resume}")
                ckpt = torch.load(resume, map_location=self.device)
                
                # 모델 가중치 로드
                self.model.load_state_dict(ckpt['model'])
                
                # 옵티마이저 및 스케줄러 상태 복원
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
                
                # EMA 모델 로드 (있는 경우)
                if self.ema and 'ema' in ckpt:
                    self.ema.module.load_state_dict(ckpt['ema'])
                
                # 훈련 상태 복원
                start_epoch = ckpt.get('epoch', 0) + 1
                best_map = ckpt.get('best_map', 0.0)

                print(f"체크포인트 로드 완료 (에포크: {start_epoch-1}, 최고 mAP: {best_map:.4f})")
            else:
                print(f"[경고] 체크포인트가 존재하지 않습니다: {resume}")

        # 학습 루프
        print(f"학습 시작 (에포크: {start_epoch+1}-{epochs})")
        no_improve = 0
        
        for epoch in range(start_epoch, epochs):
            # 훈련 모드로 설정
            self.model.train()
            epoch_loss = 0.0
            
            # 진행 상황 추적
            print(f"에포크 {epoch+1}/{epochs} 시작")
            batch_count = len(dl_train)
            log_interval = max(1, int(batch_count / 20))  # 더 자주 로깅 (5% 단위)
            
            # 배치 반복
            start_time = time.time()
            last_cache_clear = 0  # 마지막 캐시 비우기 시점
            for step, batch_data in enumerate(dl_train):
                # 진행률 표시 (%)
                progress = (step + 1) / batch_count * 100
                
                # 메모리 관리: 주기적으로 캐시 비우기 (100배치마다)
                if self.device.type == 'cuda' and (step - last_cache_clear) >= 100:
                    torch.cuda.empty_cache()
                    last_cache_clear = step
                
                # 데이터 준비
                images = batch_data["image"].to(self.device)
                targets = batch_data["target"]
                
                # 타겟 텐서가 디바이스에 올바르게 로드되었는지 확인
                for i in range(len(targets)):
                    if "boxes" in targets[i] and len(targets[i]["boxes"]) > 0:
                        if not targets[i]["boxes"].device == self.device:
                            targets[i]["boxes"] = targets[i]["boxes"].to(self.device)
                    if "labels" in targets[i]:
                        if not targets[i]["labels"].device == self.device:
                            targets[i]["labels"] = targets[i]["labels"].to(self.device)
                    if "time" in targets[i]:
                        if not targets[i]["time"].device == self.device:
                            targets[i]["time"] = targets[i]["time"].to(self.device)
                
                # AMP를 사용한 순전파 및 손실 계산
                # 최신 API 사용
                with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=amp):
                    outputs = self.model(images, targets)
                    loss = criterion(outputs, targets) / accum_steps

                # 역전파
                scaler.scale(loss).backward()

                # 그래디언트 누적 및 옵티마이저 스텝
                if (step + 1) % accum_steps == 0 or (step + 1) == batch_count:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # set_to_none=True로 메모리 사용 최적화
                    scheduler.step()
                    
                    # EMA 업데이트
                    if self.ema:
                        self.ema.update(self.model)
                
                # GPU 메모리 관리: 변수 정리
                del images
                
                # 손실 추적
                epoch_loss += loss.item() * accum_steps

                # 진행 상황 로깅 (더 자주 업데이트)
                if (step + 1) % log_interval == 0 or (step + 1) == batch_count:
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    imgs_per_sec = (step + 1) * batch / elapsed
                    eta = (batch_count - step - 1) * elapsed / (step + 1)
                    
                    # GPU 메모리 사용량 모니터링 (활성화된 경우)
                    mem_stats = ""
                    if monitor_memory and self.device.type == 'cuda':
                        # 현재 GPU 메모리 사용량 (MB 단위)
                        mem_alloc = torch.cuda.memory_allocated(self.device) / (1024**2)
                        mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
                        mem_stats = f", GPU 메모리: {mem_alloc:.0f}MB (할당) / {mem_reserved:.0f}MB (예약)"
                    
                    print(f"  [진행: {progress:.1f}%] 배치 {step+1}/{batch_count}, 손실: {loss.item():.4f}, LR: {lr:.6f}, 속도: {imgs_per_sec:.1f}장/초, ETA: {eta:.1f}초{mem_stats}")

            # 에포크 평균 손실
            avg_loss = epoch_loss / batch_count
            epoch_time = time.time() - start_time
            print(f"에포크 {epoch+1} 학습 완료 - 평균 손실: {avg_loss:.4f}, 소요 시간: {epoch_time:.1f}초")
            
            # 검증 (EMA 모델 사용, 있는 경우)
            val_model = self.ema.module if self.ema else self.model
            val_model.eval()
            
            print(f"검증 진행 중...")
            map_val = self._evaluate(val_model, dl_val)
            print(f"검증 mAP: {map_val:.4f}")
            
            # 최고 모델 저장
            if map_val > best_map:
                best_map = map_val
                print(f"새로운 최고 mAP: {best_map:.4f} - 모델 저장 중...")
                
                # 체크포인트 저장
                save_dict = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_map': best_map,
                }
                
                if self.ema:
                    save_dict['ema'] = self.ema.module.state_dict()
                
                torch.save(save_dict, best_ckpt)
                print(f"최고 모델 저장 완료: {best_ckpt}")
                
                # 조기 종료 카운터 리셋
                no_improve = 0
            else:
                no_improve += 1
                print(f"mAP 개선 없음: {no_improve}/{early_stop_patience}")
            
            # 마지막 모델 저장
            save_dict = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_map': best_map,
            }
            
            if self.ema:
                save_dict['ema'] = self.ema.module.state_dict()
            
            torch.save(save_dict, last_ckpt)
            
            # W&B 로깅
            if log_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': avg_loss,
                    'val/mAP': map_val,
                    'lr': scheduler.get_last_lr()[0]
                })
            
            # 조기 종료 확인
            if no_improve >= early_stop_patience:
                print(f"조기 종료: {early_stop_patience}회 연속 개선 없음")
                break

            # 캐시 메모리 정보 (활성화된 경우)
            if monitor_memory:
                if self.device.type == 'cuda':
                    # GPU 메모리 통계
                    mem_alloc = torch.cuda.memory_allocated(self.device) / (1024**2)
                    mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
                    print(f"GPU 메모리 사용량: {mem_alloc:.0f}MB (할당) / {mem_reserved:.0f}MB (예약)")
                
                # 데이터셋 캐시 정보
                if hasattr(ds_train, 'cache_images') and ds_train.cache_images:
                    cache_count = len(ds_train.img_cache)
                    cache_size = ds_train.current_cache_size
                    print(f"학습 데이터셋 캐시: {cache_count}개 이미지 / {cache_size:.0f}MB")
                
                if hasattr(ds_val, 'cache_images') and ds_val.cache_images:
                    cache_count = len(ds_val.img_cache)
                    cache_size = ds_val.current_cache_size
                    print(f"검증 데이터셋 캐시: {cache_count}개 이미지 / {cache_size:.0f}MB")

        # 최종 출력
        print(f"학습 완료. 최고 mAP: {best_map:.4f}, 모델 경로: {best_ckpt}")
        
        # 최고 모델 로드
        self.load(str(best_ckpt))
        
        return self

    @torch.no_grad()
    def _evaluate(self, model, dataloader):
        """
        검증 데이터에 대한 모델 평가
        
        Args:
            model: 평가할 모델
            dataloader: 검증 데이터 로더
            
        Returns:
            mAP: 평균 정밀도 점수
        """
        model.eval()
        
        # 간소화된 평가 구현
        # 실제 프로젝트에서는 COCO evaluator 사용 권장
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        start_time = time.time()
        batch_count = len(dataloader)
        log_interval = max(1, batch_count // 10)  # 10% 단위로 로깅
        
        print(f"검증 진행 중... (총 {batch_count}개 배치)")
        
        for batch_idx, batch in enumerate(dataloader):
            # 진행률 표시
            progress = (batch_idx + 1) / batch_count * 100
            
            images = batch["image"].to(self.device)
            targets = batch["target"]
            
            # 예측 수행
            outputs = model(images)
            
            # 각 이미지에 대한 평가
            for i, (output, target) in enumerate(zip(outputs, targets)):
                # 예측 박스, 점수, 클래스
                pred_boxes = output["boxes"][i]
                pred_scores = output["scores"][i]
                pred_labels = output["labels"][i]
                
                # 타겟 박스, 클래스
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]
                
                # 임계값 이상 예측만 사용
                keep = pred_scores > 0.5
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                
                # 간단한 매칭 (실제로는 더 정교한 평가 필요)
                matched_gt = set()
                tp = 0
                fp = 0
                
                for pb, pl in zip(pred_boxes, pred_labels):
                    best_iou = 0.5  # IoU 임계값
                    best_gt = -1
                    
                    for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if j in matched_gt or pl != gl:
                            continue
                        
                        # IoU 계산 (간소화)
                        iou = self._box_iou(pb.cpu(), gb.cpu())
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = j
                    
                    if best_gt >= 0:
                        tp += 1
                        matched_gt.add(best_gt)
                    else:
                        fp += 1
                
                total_tp += tp
                total_fp += fp
                total_gt += len(gt_boxes)
            
            # 진행 상황 로깅
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == batch_count:
                elapsed = time.time() - start_time
                imgs_per_sec = (batch_idx + 1) * dataloader.batch_size / elapsed
                eta = (batch_count - batch_idx - 1) * elapsed / (batch_idx + 1)
                print(f"  [검증: {progress:.1f}%] 배치 {batch_idx+1}/{batch_count}, 속도: {imgs_per_sec:.1f}장/초, ETA: {eta:.1f}초")
        
        # 정밀도, 재현율, mAP 계산
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        # 간소화된 mAP (F1을 대신 사용)
        map_val = f1
        
        val_time = time.time() - start_time
        print(f"검증 완료 (소요 시간: {val_time:.1f}초)")
        print(f"검증 지표 - Precision: {precision:.4f}, Recall: {recall:.4f}, F1/mAP: {map_val:.4f}")
        
        return map_val
    
    def _box_iou(self, box1, box2):
        """
        두 박스 간의 IoU 계산 (간소화된 버전)
        
        Args:
            box1: 첫 번째 박스 [x1, y1, x2, y2]
            box2: 두 번째 박스 [x1, y1, x2, y2]
            
        Returns:
            iou: IoU 값
        """
        # 교차 영역 계산
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # 각 박스의 면적
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # IoU 계산
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / max(union_area, 1e-6)
        
        return iou

    @torch.no_grad()
    def predict(self, img, conf_thresh=0.25):
        """
        이미지에서 객체 감지 수행
        
        Args:
            img: OpenCV 이미지 (np.ndarray) 또는 이미지 텐서 (torch.Tensor)
            conf_thresh: 신뢰도 임계값
            
        Returns:
            list: [[x1,y1,x2,y2,score,class_id], ...] 형식의 검출 결과
        """
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 이미지 전처리
        if isinstance(img, np.ndarray):
            # OpenCV 이미지 (BGR)를 RGB로 변환
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        elif isinstance(img, torch.Tensor):
            # 이미 텐서인 경우 (가정: [C,H,W], 범위 [0,1])
            img_tensor = img
        else:
            raise ValueError("지원되지 않는 이미지 형식입니다.")
        
        # 차원 확인 및 배치 차원 추가
        if img_tensor.dim() == 3:  # [C,H,W]
            img_tensor = img_tensor.unsqueeze(0)  # [1,C,H,W]
        
        # 디바이스로 이동
        img_tensor = img_tensor.to(self.device)

        # 추론 수행
        outputs = self.model(img_tensor)
        
        # 결과 후처리
        boxes = outputs['boxes'][0]  # [N,4], 첫 번째 이미지의 결과만 사용
        scores = outputs['scores'][0]  # [N]
        labels = outputs['labels'][0]  # [N]
        
        # 임계값 필터링
        keep = scores > conf_thresh
        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        labels = labels[keep].cpu().numpy()
        
        # 결과 형식 변환: [[x1,y1,x2,y2,score,class_id], ...]
        results = []
        for box, score, label in zip(boxes, scores, labels):
            results.append([*box, score, label])
        
        return results

    def save(self, path):
        """
        모델 가중치 저장
        
        Args:
            path: 저장 경로
        """
        # 저장할 가중치 선택 (EMA가 있으면 EMA 사용)
        weights = self.ema.module.state_dict() if self.ema else self.model.state_dict()
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 가중치 저장
        torch.save(weights, path)
        print(f"모델 가중치 저장 완료: {path}")

    def load(self, path):
        """
        모델 가중치 로드
        
        Args:
            path: 가중치 파일 경로
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {path}")
        
        # 가중치 로드
        if path.lower().endswith('.pt') or path.lower().endswith('.pth'):
            # 모델 가중치만 포함하는 파일
            try:
                state_dict = torch.load(path, map_location=self.device)
                
                # 체크포인트 형식 확인
                if isinstance(state_dict, dict) and 'model' in state_dict:
                    # 학습 체크포인트인 경우
                    state_dict = state_dict['model']
                
                self.model.load_state_dict(state_dict)
                print(f"모델 가중치 로드 완료: {path}")
            except Exception as e:
                raise RuntimeError(f"가중치 로드 중 오류 발생: {e}")
        else:
            raise ValueError("지원되지 않는 가중치 파일 형식입니다. .pt 또는 .pth 파일이 필요합니다.")
        
        # 모델을 디바이스로 이동
        self.model.to(self.device)
        
        return self
