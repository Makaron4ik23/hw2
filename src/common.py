"""Shared utilities for HW2 (VisDrone-VID multi-object detection).

Every notebook imports this file, so data preparation, evaluation and the
submission format are identical for all detectors.

Sections:
  1. Paths, constants, seeding
  2. Reading labels / ignore regions
  3. Building the training copy (frame subsampling + masking ignored regions)
  4. COCO evaluation on the raw validation frames (ignored regions = iscrowd)
  5. Prediction helpers (Ultralytics, torchvision) + speed measurement
  6. Torchvision dataset and training loop
  7. Visualisation of failure cases
  8. Submission writer
"""
import contextlib
import io
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CLASSES = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']
NC = len(CLASSES)
SEED = 42


# --------------------------------------------------------------------------
# 1. Paths, constants, seeding
# --------------------------------------------------------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_data_root():
    """Folder that contains train/, val/, test/ and test_images.csv.

    Set the DATA_ROOT environment variable to override (Colab / local runs).
    """
    if os.environ.get('DATA_ROOT'):
        return Path(os.environ['DATA_ROOT'])
    for pattern in ['*', '*/*', '*/*/*']:
        hits = sorted(Path('/kaggle/input').glob(f'{pattern}/test_images.csv'))
        if hits:
            return hits[0].parent
    raise FileNotFoundError('Dataset not found: attach the competition data or set DATA_ROOT')


def list_frames(split_dir, step=1):
    """Every `step`-th frame of every clip: list of (clip, image_path)."""
    out = []
    for clip_dir in sorted(p for p in (Path(split_dir) / 'images').iterdir() if p.is_dir()):
        frames = sorted(clip_dir.glob('*.jpg'))
        out += [(clip_dir.name, f) for f in frames[::step]]
    return out


def frame_key(clip, path):
    """Unique key of an original frame, e.g. 'clip_003/0000017.jpg'."""
    return f'{clip}/{Path(path).name}'


# --------------------------------------------------------------------------
# 2. Labels and ignored regions
# --------------------------------------------------------------------------
def _read_rows(path, ncols):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, ncols), dtype=np.float32)
    arr = np.loadtxt(path, dtype=np.float32, ndmin=2)
    return arr.reshape(-1, ncols)


def cxcywh_to_xyxy(a, w, h):
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def read_yolo_labels(label_path, w, h):
    """-> (boxes xyxy in pixels, class ids 0..7)."""
    rows = _read_rows(label_path, 5)
    return cxcywh_to_xyxy(rows[:, 1:], w, h), rows[:, 0].astype(np.int64)


def read_ignore(ignore_path, w, h):
    """-> ignored regions as xyxy pixels."""
    return cxcywh_to_xyxy(_read_rows(ignore_path, 4), w, h)


def label_path_for(root, split, clip, img_path):
    return Path(root) / split / 'labels' / clip / (Path(img_path).stem + '.txt')


def ignore_path_for(root, split, clip, img_path):
    return Path(root) / split / 'ignore' / clip / (Path(img_path).stem + '.txt')


# --------------------------------------------------------------------------
# 3. Training copy: subsample frames and grey out ignored regions
# --------------------------------------------------------------------------
FILL = (114, 114, 114)  # same grey Ultralytics uses for letterbox padding


def mask_ignored(img, ignore_xyxy, gt_xyxy):
    """Fill ignored regions with grey, but keep pixels of real objects.

    Ignored regions contain unlabelled crowds / parking lots; leaving them
    as-is would teach the detector that these objects are background.
    """
    if len(ignore_xyxy) == 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for x1, y1, x2, y2 in ignore_xyxy.round().astype(int):
        out[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = FILL
    for x1, y1, x2, y2 in gt_xyxy.round().astype(int):  # restore labelled objects
        out[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    return out


def prepare_training_copy(root, out_dir, step_train=5, step_val=5, workers=8):
    """Write a flat YOLO-format copy: out/{train,val}/{images,labels}/<clip>_<frame>.*

    Returns path to data.yaml. Re-running is a no-op if the copy exists.
    """
    root, out_dir = Path(root), Path(out_dir)
    yaml_path = out_dir / 'data.yaml'
    if yaml_path.exists():
        print('Training copy already exists:', out_dir)
        return yaml_path

    rows = []
    for split, step in [('train', step_train), ('val', step_val)]:
        (out_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (out_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        frames = list_frames(root / split, step)

        def work(item, split=split):
            clip, p = item
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            lbl = label_path_for(root, split, clip, p)
            gt, _ = read_yolo_labels(lbl, w, h)
            ign = read_ignore(ignore_path_for(root, split, clip, p), w, h)
            name = f'{clip}_{p.stem}'
            cv2.imwrite(str(out_dir / split / 'images' / f'{name}.jpg'), mask_ignored(img, ign, gt),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            dst = out_dir / split / 'labels' / f'{name}.txt'
            if lbl.exists():
                shutil.copy(lbl, dst)
            else:
                dst.write_text('')
            return dict(split=split, clip=clip, frame=p.name, orig=str(p),
                        image=str(out_dir / split / 'images' / f'{name}.jpg'), n_gt=len(gt), n_ignore=len(ign))

        with ThreadPoolExecutor(workers) as ex:
            rows += list(ex.map(work, frames))
        print(f'{split}: {len(frames)} frames (every {step}th)')

    pd.DataFrame(rows).to_csv(out_dir / 'manifest.csv', index=False)
    yaml_path.write_text(
        f'path: {out_dir}\ntrain: train/images\nval: val/images\n'
        f'nc: {NC}\nnames: {json.dumps(CLASSES)}\n')
    return yaml_path


# --------------------------------------------------------------------------
# 4. COCO evaluation on raw validation frames
# --------------------------------------------------------------------------
class CocoEvaluator:
    """COCO mAP on the original val frames.

    Ignored regions are iscrowd=1 annotations in annotations_coco.json, so
    pycocotools does not count detections inside them as false positives.
    maxDets is raised to 500 (the leaderboard scores up to 500 boxes / image).
    """

    MAX_DETS = [1, 100, 500]

    def __init__(self, root, frames):
        from pycocotools.coco import COCO
        with contextlib.redirect_stdout(io.StringIO()):
            self.gt = COCO(str(Path(root) / 'val' / 'annotations_coco.json'))
        by_key = {}
        for img in self.gt.dataset['images']:
            fn = img['file_name'].replace('\\', '/')
            parts = fn.split('/')
            clip = parts[-2] if len(parts) >= 2 else str(img.get('video'))
            by_key[f'{clip}/{parts[-1]}'] = img['id']
            if 'video' in img:  # second key in case file_name has no clip folder
                by_key.setdefault(f"{img['video']}/{parts[-1]}", img['id'])
        self.key_to_id = by_key
        self.frames = frames
        missing = [frame_key(c, p) for c, p in frames if frame_key(c, p) not in by_key]
        assert not missing, f'{len(missing)} val frames not in COCO json, e.g. {missing[:3]}'
        self.img_ids = [by_key[frame_key(c, p)] for c, p in frames]

    def to_coco_results(self, preds):
        """preds: {frame_key: (xyxy [N,4], scores [N], labels0 [N])}"""
        res = []
        for key, (b, s, l) in preds.items():
            if key not in self.key_to_id:
                continue
            iid = self.key_to_id[key]
            for (x1, y1, x2, y2), sc, lb in zip(np.asarray(b), np.asarray(s), np.asarray(l)):
                res.append(dict(image_id=iid, category_id=int(lb) + 1, score=float(sc),
                                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]))
        return res

    def evaluate(self, preds, verbose=False):
        from pycocotools.cocoeval import COCOeval
        res = self.to_coco_results(preds)
        if not res:
            return dict(mAP=0.0, AP50=0.0, AP75=0.0, AP_small=0.0, AP_medium=0.0, AP_large=0.0, AR500=0.0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dt = self.gt.loadRes(res)
            ev = COCOeval(self.gt, dt, 'bbox')
            ev.params.imgIds = self.img_ids
            ev.params.maxDets = self.MAX_DETS
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        if verbose:
            print(buf.getvalue())
        s = ev.stats
        out = dict(mAP=s[0], AP50=s[1], AP75=s[2], AP_small=s[3], AP_medium=s[4], AP_large=s[5], AR500=s[8])
        # per-class AP (IoU .50:.95, all areas, maxDets=500)
        prec = ev.eval['precision'][:, :, :, 0, 2]  # [T, R, K]
        for k, cat_id in enumerate(ev.params.catIds):
            p = prec[:, :, k]
            p = p[p > -1]
            out[f'AP_{CLASSES[cat_id - 1]}'] = float(p.mean()) if p.size else float('nan')
        self.last_eval = ev
        return {k: round(float(v), 4) for k, v in out.items()}

    SIZE_BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 100000)]  # sqrt(box area) in original pixels

    def evaluate_by_size(self, preds):
        """AP (IoU .5:.95) and AP50 / recall for finer size bins than COCO's small/medium/large."""
        from pycocotools.cocoeval import COCOeval
        res = self.to_coco_results(preds)
        if not res:
            return pd.DataFrame()
        labels = ['all'] + [f'{a}-{b if b < 1e5 else "inf"}px' for a, b in self.SIZE_BINS]
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(self.gt, self.gt.loadRes(res), 'bbox')
            ev.params.imgIds = self.img_ids
            ev.params.maxDets = self.MAX_DETS
            ev.params.areaRng = [[0, 1e10]] + [[a * a, b * b] for a, b in self.SIZE_BINS]
            ev.params.areaRngLbl = labels
            ev.evaluate()
            ev.accumulate()
        prec = ev.eval['precision'][..., 2]  # [T, R, K, A]
        rec = ev.eval['recall'][..., 2]  # [T, K, A]
        anns = [a for a in self.gt.loadAnns(self.gt.getAnnIds(imgIds=self.img_ids)) if not a.get('iscrowd', 0)]
        areas = np.array([a['bbox'][2] * a['bbox'][3] for a in anns])
        rows = []
        for i, lab in enumerate(labels):
            lo, hi = ev.params.areaRng[i]
            mean = lambda x: float(x[x > -1].mean()) if (x > -1).any() else float('nan')
            rows.append(dict(size=lab, n_gt=int(((areas >= lo) & (areas < hi)).sum()), AP=round(mean(prec[:, :, :, i]), 4),
                             AP50=round(mean(prec[0, :, :, i]), 4), recall50=round(mean(rec[0, :, i]), 4)))
        return pd.DataFrame(rows)


def log_result(row, path='results/metrics.csv'):
    """Append one experiment's metrics to a shared CSV (used for the report tables)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------
# 5. Prediction helpers
# --------------------------------------------------------------------------
def predict_ultralytics(model, frames, imgsz, conf=0.001, iou=0.6, max_det=500,
                        augment=False, batch=8, half=True):
    """Run an Ultralytics model (YOLO or RT-DETR) on raw frames.

    Returns ({frame_key: (xyxy, scores, labels0)}, images_per_second).
    """
    import torch
    preds, n, t0 = {}, 0, None
    paths = [str(p) for _, p in frames]
    keys = [frame_key(c, p) for c, p in frames]
    half = half and torch.cuda.is_available()
    for i in range(0, len(paths), batch):
        if i == batch:  # start the clock after the first (warm-up) batch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0, n = time.time(), 0
        results = model.predict(paths[i:i + batch], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                                augment=augment, half=half, verbose=False)
        for k, r in zip(keys[i:i + batch], results):
            b = r.boxes
            preds[k] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
        n += len(results)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fps = n / (time.time() - t0) if t0 else float('nan')
    return preds, round(fps, 2)


def predict_torchvision(model, frames, device, batch=4, score_thresh=0.001):
    """Run a torchvision detector (labels 1..8 inside the model) on raw frames."""
    import torch
    model.eval().to(device)
    preds, n, t0 = {}, 0, None
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        imgs = [torch.from_numpy(cv2.imread(str(p))[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(device)
                for _, p in chunk]
        if i == batch:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0, n = time.time(), 0
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == 'cuda'):
            out = model(imgs)
        for (c, p), o in zip(chunk, out):
            keep = o['scores'] >= score_thresh
            preds[frame_key(c, p)] = (o['boxes'][keep].float().cpu().numpy(),
                                      o['scores'][keep].float().cpu().numpy(),
                                      o['labels'][keep].cpu().numpy() - 1)
        n += len(chunk)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fps = n / (time.time() - t0) if t0 else float('nan')
    return preds, round(fps, 2)


def _nms_topk(boxes, scores, labels, iou, max_det):
    import torch
    from torchvision.ops import batched_nms
    if len(boxes) == 0:
        return boxes, scores, labels
    keep = batched_nms(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(scores, dtype=torch.float32),
                       torch.as_tensor(labels), iou).numpy()[:max_det]
    return boxes[keep], scores[keep], labels[keep]


def predict_ultralytics_tiled(model, frames, imgsz, grid=(2, 2), overlap=0.2, include_full=True,
                              conf=0.001, iou=0.6, max_det=500, half=True):
    """Slicing inference (idea from SAHI, Akyon et al. 2022, re-implemented here).

    Each frame is cut into grid[0] x grid[1] overlapping tiles; every tile is
    resized to `imgsz`, so small objects are seen at higher resolution. Tile
    detections (+ optionally the full frame) are merged with class-wise NMS.
    """
    import torch
    half = half and torch.cuda.is_available()
    preds, t0 = {}, time.time()
    gx, gy = grid
    for c, p in frames:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        tw, th = int(np.ceil(w / (gx - overlap * (gx - 1)))), int(np.ceil(h / (gy - overlap * (gy - 1))))
        xs = np.linspace(0, w - tw, gx).astype(int) if gx > 1 else [0]
        ys = np.linspace(0, h - th, gy).astype(int) if gy > 1 else [0]
        crops, offs = [], []
        for y in ys:
            for x in xs:
                crops.append(img[y:y + th, x:x + tw])
                offs.append((x, y))
        if include_full:
            crops.append(img)
            offs.append((0, 0))
        results = model.predict(crops, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, half=half, verbose=False)
        B, S, L = [], [], []
        for (ox, oy), r in zip(offs, results):
            b = r.boxes.xyxy.cpu().numpy()
            B.append(b + np.array([ox, oy, ox, oy], dtype=np.float32))
            S.append(r.boxes.conf.cpu().numpy())
            L.append(r.boxes.cls.cpu().numpy().astype(int))
        preds[frame_key(c, p)] = _nms_topk(np.concatenate(B), np.concatenate(S), np.concatenate(L), iou, max_det)
    fps = len(frames) / (time.time() - t0)
    return preds, round(fps, 2)


def temporal_rescore(preds, frames, alpha=0.7, iou_thr=0.3, window=1):
    """Bonus (data temporality): smooth detection scores along time.

    For every box in frame t we look for the best same-class box (IoU >= iou_thr)
    in frames t-window..t+window of the same clip. New score =
    alpha * own score + (1 - alpha) * mean(best matched neighbour score, 0 if none).
    Objects that persist get their score kept, flickering false positives are
    pushed down. Idea follows Seq-NMS (Han et al., 2016) in a simplified form.
    `frames` must be consecutive frames in time order.
    """
    by_clip = {}
    for c, p in frames:
        by_clip.setdefault(c, []).append(frame_key(c, p))
    out = {}
    for clip, keys in by_clip.items():
        keys = sorted(keys)
        for t, k in enumerate(keys):
            b, s, l = preds.get(k, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
            if len(b) == 0:
                out[k] = (b, s, l)
                continue
            neigh = [keys[j] for j in range(max(0, t - window), min(len(keys), t + window + 1)) if j != t]
            support = np.zeros((len(b), max(1, len(neigh))), dtype=np.float32)
            for j, nk in enumerate(neigh):
                nb, ns, nl = preds.get(nk, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
                if len(nb) == 0:
                    continue
                iou = box_iou_np(b, nb) * (l[:, None] == nl[None, :])
                best = iou.argmax(1)
                support[:, j] = np.where(iou.max(1) >= iou_thr, ns[best], 0.0)
            out[k] = (b, alpha * s + (1 - alpha) * support.mean(1), l)
    return out


# --------------------------------------------------------------------------
# 6. Torchvision dataset + training loop
# --------------------------------------------------------------------------
def build_torch_dataset(manifest_csv, split, train):
    import torch

    class DetDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            r = self.df.iloc[i]
            img = cv2.imread(r.image)[:, :, ::-1]
            h, w = img.shape[:2]
            lbl = Path(r.image.replace('/images/', '/labels/')).with_suffix('.txt')
            boxes, cls = read_yolo_labels(lbl, w, h)
            boxes = boxes.clip([0, 0, 0, 0], [w, h, w, h])
            ok = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
            boxes, cls = boxes[ok], cls[ok]
            if train and random.random() < 0.5:  # horizontal flip
                img = img[:, ::-1]
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255)
            target = dict(boxes=torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                          labels=torch.as_tensor(cls + 1, dtype=torch.int64))  # 0 = background
            return img, target

    df = pd.read_csv(manifest_csv)
    return DetDataset(df[df.split == split])


def collate(batch):
    return tuple(zip(*batch))


def train_torchvision(model, train_ds, evaluator, val_frames, device, epochs=4, lr=0.005, batch=4,
                      workers=2, out_dir='runs/tv', log_every=200, wandb_run=None):
    """SGD + linear warm-up + cosine decay, AMP. Validates (COCO mAP) after every epoch.

    Returns history (list of dicts) and saves the best checkpoint to out_dir/best.pt.
    """
    import math
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(SEED)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                                         collate_fn=collate, generator=g, drop_last=True)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total = epochs * len(loader)
    warm = max(1, min(500, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    history, best, it, t_start = [], -1, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for imgs, targets in loader:
            imgs = [x.to(device, non_blocking=True) for x in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                print('non-finite loss, skipping batch')
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            it += 1
            if it % log_every == 0:
                print(f'ep {ep} it {it}/{total} loss {run_loss / log_every:.3f} '
                      + ' '.join(f'{k}={v.item():.3f}' for k, v in losses.items()))
                if wandb_run:
                    wandb_run.log({'train/loss': run_loss / log_every, **{f'train/{k}': v.item() for k, v in losses.items()}}, step=it)
                run_loss = 0.0
        preds, fps = predict_torchvision(model, val_frames, device)
        m = evaluator.evaluate(preds)
        row = dict(epoch=ep, minutes=round((time.time() - t_start) / 60, 1), fps=fps,
                   **{k: m[k] for k in ['mAP', 'AP50', 'AP_small']})
        history.append(row)
        print('VAL', row)
        if wandb_run:
            wandb_run.log({f'val/{k}': v for k, v in row.items()}, step=it)
        if m['mAP'] > best:
            best = m['mAP']
            torch.save(model.state_dict(), out_dir / 'best.pt')
    pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device))
    return history


# --------------------------------------------------------------------------
# 7. Failure-case visualisation
# --------------------------------------------------------------------------
def box_iou_np(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def frame_errors(root, frames, preds, score_thr=0.3, iou_thr=0.5):
    """Per-frame counts of TP / missed GT / false positives / wrong class (for picking failure cases)."""
    rows = []
    for c, p in frames:
        img_h, img_w = cv2.imread(str(p)).shape[:2]
        gt, gcls = read_yolo_labels(label_path_for(root, 'val', c, p), img_w, img_h)
        ign = read_ignore(ignore_path_for(root, 'val', c, p), img_w, img_h)
        b, s, l = preds.get(frame_key(c, p), (np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
        keep = s >= score_thr
        b, l = b[keep], l[keep]
        iou = box_iou_np(gt, b)
        matched_gt = iou.max(1) >= iou_thr if len(b) else np.zeros(len(gt), bool)
        matched_dt = iou.max(0) >= iou_thr if len(gt) else np.zeros(len(b), bool)
        in_ign = (box_iou_np(b, ign).max(1) > 0) if len(ign) and len(b) else np.zeros(len(b), bool)
        wrong_cls = 0
        if len(gt) and len(b):
            j = iou.argmax(1)
            wrong_cls = int(((iou.max(1) >= iou_thr) & (l[j] != gcls)).sum())
        rows.append(dict(key=frame_key(c, p), path=str(p), clip=c, n_gt=len(gt), tp=int(matched_gt.sum()),
                         missed=int((~matched_gt).sum()), fp=int((~matched_dt & ~in_ign).sum()), wrong_cls=wrong_cls))
    return pd.DataFrame(rows)


def show_frame(root, path, clip, preds, score_thr=0.3, ax=None, title=''):
    """GT green, detections red (class+score); ignored regions hatched grey."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    img = cv2.imread(str(path))[:, :, ::-1]
    h, w = img.shape[:2]
    gt, gcls = read_yolo_labels(label_path_for(root, 'val', clip, path), w, h)
    ign = read_ignore(ignore_path_for(root, 'val', clip, path), w, h)
    ax = ax or plt.subplots(figsize=(14, 8))[1]
    ax.imshow(img)
    for x1, y1, x2, y2 in ign:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, hatch='//', ec='grey', lw=1))
    for (x1, y1, x2, y2), c in zip(gt, gcls):
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec='lime', lw=1))
    if preds is not None:
        b, s, l = preds.get(frame_key(clip, path), (np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
        for (x1, y1, x2, y2), sc, lb in zip(b, s, l):
            if sc < score_thr:
                continue
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec='red', lw=1))
            ax.text(x1, y1 - 2, f'{CLASSES[int(lb)]} {sc:.2f}', color='red', fontsize=6)
    ax.set_title(title)
    ax.axis('off')
    return ax


# --------------------------------------------------------------------------
# 8. Submission
# --------------------------------------------------------------------------
def write_submission(preds, root, out_csv='submission.csv', max_det=500, min_score=0.01):
    """preds keyed by frame_key of test frames -> Kaggle CSV (class score x y w h, pixels)."""
    test = pd.read_csv(Path(root) / 'test_images.csv')
    lines = []
    n_boxes = 0
    for r in test.itertuples():
        key = frame_key(r.clip, r.file_name)
        parts = []
        if key in preds:
            b, s, l = preds[key]
            order = np.argsort(-np.asarray(s))[:max_det]
            for i in order:
                if s[i] < min_score:
                    continue
                x1, y1, x2, y2 = b[i]
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(r.width), x2), min(float(r.height), y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                parts.append(f'{int(l[i])} {s[i]:.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}')
        n_boxes += len(parts)
        lines.append((r.image_id, ' '.join(parts)))
    sub = pd.DataFrame(lines, columns=['image_id', 'PredictionString'])
    sub.to_csv(out_csv, index=False)
    print(f'{out_csv}: {len(sub)} rows, {n_boxes} boxes, {n_boxes / len(sub):.1f} per image')
    return sub


def test_frames(root):
    """All test frames as (clip, path), in test_images.csv order."""
    test = pd.read_csv(Path(root) / 'test_images.csv')
    return [(r.clip, Path(root) / r.file_name) for r in test.itertuples()]
