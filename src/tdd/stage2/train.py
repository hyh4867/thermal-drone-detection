"""Stage 2 / Q3 training runs (pre-registration section 5/Q3). At most 4 runs in total.

R1  small-target augmentation, Anti-UAV train only (no Svanström data):
    PYTHONPATH=src .venv/bin/python -m tdd.stage2.train --run r1 --epochs 2 --small-aug 0.5
R2  few-shot target-domain fine-tuning (6 Anti-UAV + 2 Svanström dev_a crops per batch):
    PYTHONPATH=src .venv/bin/python -m tdd.stage2.train --run r2_n4 --epochs 1 --svan-n 4 --init <ckpt>

Low-memory settings of configs/full.yaml: no worker processes, micro-batches of 2 with
gradient accumulation (effective batch 8), evaluation one image at a time. The batch
order is a pure function of (seed, epoch), so a killed run resumes at the exact
iteration from runs/stage2/<run>/state.pt (saved every 250 iterations and per epoch).
Wall-clock limit (default 150 min, all processes of the run added up): when it is
exceeded the run stops at the last saved weights and says so in its log.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import cv2
import numpy as np
import torch

from ..common import ROOT, load_config, read_json, run_metadata, set_seed, write_json
from ..evaluate import ap_on, load_image
from ..metrics import size_of
from ..model import build_model, make_predict_fn, set_input_size
from ..train import CropDataset, _empty_cache, _mem_stats
from .data import MAN, frame_set
from .split import load_split

SAVE_EVERY = 250


class SmallTargetDataset(CropDataset):
    """Stage 1 crops plus the R1 small-target augmentation.

    With probability ``p_small`` (frames with a target only) the frame is shrunk so
    that the target's sqrt(w*h) becomes uniform in [4, 12] px. For half of those the
    low-resolution sensor is imitated: shrink to 1/k of the final size (k in 2, 3, 4),
    enlarge k times with bilinear interpolation (blur), add Gaussian noise (sigma 0-4
    grey levels). Otherwise the stage 1 scale jitter applies unchanged."""

    def __init__(self, frames, tc, seed, p_small):
        super().__init__(frames, tc, seed)
        self.p_small = p_small

    def __getitem__(self, idx):
        rng = random.Random(hash((self.seed, self.epoch, idx, 7)) & 0xFFFFFFFF)
        fr = self.frames[idx]
        if not fr["gt"] or rng.random() >= self.p_small:
            return super().__getitem__(idx)
        g = load_image(fr["image"])
        boxes = np.asarray(fr["gt"], np.float32).reshape(-1, 4)
        size = size_of(boxes[0])
        s = min(1.0, rng.uniform(4.0, 12.0) / max(size, 1e-6))
        if rng.random() < 0.5:
            k = rng.choice([2, 3, 4])
            small = cv2.resize(g, None, fx=s / k, fy=s / k, interpolation=cv2.INTER_AREA)
            tw, th = max(1, round(g.shape[1] * s)), max(1, round(g.shape[0] * s))
            g2 = cv2.resize(small, (tw, th), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            sigma = rng.uniform(0.0, 4.0)
            noise = np.random.default_rng(rng.getrandbits(32)).normal(0.0, sigma, g2.shape)
            g2 = np.clip(g2 + noise, 0, 255).astype(np.uint8)
        else:
            g2 = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        sx, sy = g2.shape[1] / g.shape[1], g2.shape[0] / g.shape[0]
        boxes = boxes * np.array([sx, sy, sx, sy], np.float32)
        return crop_and_flip(g2, boxes, self.tc, rng)


def crop_and_flip(g, boxes, tc, rng, min_side=2.0):
    """The crop / pad / flip part of the stage 1 CropDataset, for an already scaled image."""
    H, W = g.shape
    c = tc["crop"]
    if H < c or W < c:
        pad = np.full((max(H, c), max(W, c)), int(np.median(g)), np.uint8)
        pad[:H, :W] = g
        g, (H, W) = pad, pad.shape
    if len(boxes) and rng.random() < tc["crop_contains_target_p"]:
        x1, y1, x2, y2 = boxes[0]
        lo_x, hi_x = max(0, int(math.ceil(x2)) - c), min(int(x1), W - c)
        lo_y, hi_y = max(0, int(math.ceil(y2)) - c), min(int(y1), H - c)
        cx = rng.randint(lo_x, hi_x) if hi_x >= lo_x else rng.randint(0, W - c)
        cy = rng.randint(lo_y, hi_y) if hi_y >= lo_y else rng.randint(0, H - c)
    else:
        cx, cy = rng.randint(0, W - c), rng.randint(0, H - c)
    g = g[cy:cy + c, cx:cx + c]
    if len(boxes):
        b = boxes - np.array([cx, cy, cx, cy], np.float32)
        area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        b = np.clip(b, 0, c)
        area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = (area1 >= 0.5 * area0) & (b[:, 2] - b[:, 0] >= min_side) & (b[:, 3] - b[:, 1] >= min_side)
        boxes = b[keep]
    if rng.random() < tc["hflip_p"]:
        g = g[:, ::-1]
        if len(boxes):
            boxes = np.stack([c - boxes[:, 2], boxes[:, 1], c - boxes[:, 0], boxes[:, 3]], 1)
    img = torch.from_numpy(np.ascontiguousarray(g)).float().div_(255.0).unsqueeze(0).expand(3, -1, -1).contiguous()
    target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
              "labels": torch.ones(len(boxes), dtype=torch.int64)}
    return img, target


class SvanAdaptDataset:
    """Svanström dev_a frames for R2: upscaled by ``scale`` (as at inference), 320x320 crop.
    Only DRONE boxes are targets; bird / airplane / helicopter frames are negatives."""

    def __init__(self, frames, tc, seed, scale):
        self.frames, self.tc, self.seed, self.scale = frames, tc, seed, scale

    def __len__(self):
        return len(self.frames)

    def get(self, idx, epoch, it):
        rng = random.Random(hash((self.seed, epoch, it, idx, 11)) & 0xFFFFFFFF)
        fr = self.frames[idx]
        g = cv2.resize(load_image(fr["image"]), None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        boxes = np.asarray(fr["gt"] if fr["cls"] == "DRONE" else [], np.float32).reshape(-1, 4) * self.scale
        return crop_and_flip(g, boxes, self.tc, rng)


class CstAdaptDataset:
    """Stage 3 track 2: CST Anti-UAV train frames at native size, 320x320 crop. Every box is a drone."""

    def __init__(self, frames, tc, seed):
        self.frames, self.tc, self.seed = frames, tc, seed

    def __len__(self):
        return len(self.frames)

    def get(self, idx, epoch, it):
        rng = random.Random(hash((self.seed, epoch, it, idx, 13)) & 0xFFFFFFFF)
        fr = self.frames[idx]
        return crop_and_flip(load_image(fr["image"]), np.asarray(fr["gt"], np.float32).reshape(-1, 4), self.tc, rng)


def pick_adapt_videos(n_drone: int, seed: int = 0):
    """First ``n_drone`` dev_a drone videos and round(n_drone * 49/37) dev_a non-drone
    videos of a fixed-seed order (pre-registered)."""
    split = read_json(ROOT / "docs" / "splits" / "svanstrom_stage2_split.json")["videos"]
    drone = sorted(v["video"] for v in split if v["split"] == "dev_a" and v["cls"] == "DRONE")
    other = sorted(v["video"] for v in split if v["split"] == "dev_a" and v["cls"] != "DRONE")
    rng = np.random.default_rng(seed)
    drone = [drone[i] for i in rng.permutation(len(drone))]
    other = [other[i] for i in rng.permutation(len(other))]
    n_other = min(len(other), int(round(n_drone * len(other) / len(drone))))
    return drone[:n_drone], other[:n_other]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--init", default="runs/full_frcnn/best.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--small-aug", type=float, default=0.0, help="R1: share of target frames given the small-target augmentation")
    ap.add_argument("--svan-n", type=int, default=0, help="R2: number of dev_a drone videos mixed in (0 = none)")
    ap.add_argument("--svan-per-batch", type=int, default=2)
    ap.add_argument("--svan-scale", type=float, default=2.0)
    ap.add_argument("--time-limit-min", type=float, default=150.0)
    ap.add_argument("--stage", type=int, default=2, help="output folders runs/stage<N>, results/stage<N>")
    ap.add_argument("--anchors", default="", help="stage 3: comma-separated anchor sizes, one per FPN level (default: config)")
    ap.add_argument("--cst-split", default="", help="stage 3 track 2: cst_tr_5 or cst_tr_20 frames mixed in (never used in track 1)")
    ap.add_argument("--cst-per-batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None, help="stage 3: overrides the config seed")
    ap.add_argument("--max-iters", type=int, default=0, help="smoke test only (does not count as a run below 100)")
    ap.add_argument("--smoke-persist", action="store_true", help="smoke test of the checkpoint / resume path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.anchors:
        cfg["model"]["anchor_sizes"] = [int(v) for v in args.anchors.split(",")]
    assert not (args.svan_n and args.cst_split), "one adaptation source per run"
    set_seed(cfg["seed"])
    persist = not args.max_iters or args.smoke_persist
    tc, mc = cfg["train"], cfg["model"]
    device = torch.device(tc["device"] if torch.backends.mps.is_available() else "cpu")
    run_dir = ROOT / "runs" / f"stage{args.stage}" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    res_path = ROOT / "results" / f"stage{args.stage}" / f"train_{args.run}.json"

    train = read_json(MAN / "antiuav_train.json")
    val = read_json(MAN / "antiuav_val.json")
    if args.max_iters:
        val = val[:16]
    ds = SmallTargetDataset(train, tc, cfg["seed"], args.small_aug) if args.small_aug > 0 else CropDataset(train, tc, cfg["seed"])
    svan, adapt = None, None
    bs, micro = tc["batch_size"], int(tc.get("micro_batch_size", tc["batch_size"]))
    n_auv = bs
    if args.svan_n:
        dv, ov = pick_adapt_videos(args.svan_n)
        keep = set(dv) | set(ov)
        split = load_split()
        frames = [f for f in frame_set("svan_dev_a") if f["video"] in keep]
        assert all(split[f["video"]] == "dev_a" for f in frames), "only dev_a videos may be trained on"
        svan = SvanAdaptDataset(frames, tc, cfg["seed"], args.svan_scale)
        n_auv = bs - args.svan_per_batch
        adapt = {"drone_videos": dv, "non_drone_videos": ov, "frames": len(frames),
                 "drone_frames_with_box": sum(1 for f in frames if f["cls"] == "DRONE" and f["gt"]),
                 "per_batch": args.svan_per_batch, "scale": args.svan_scale}
    if args.cst_split:
        from ..stage3.data import frame_set as frame_set3
        frames = frame_set3(args.cst_split)
        svan = CstAdaptDataset(frames, tc, cfg["seed"])
        n_auv = bs - args.cst_per_batch
        args.svan_per_batch = args.cst_per_batch
        adapt = {"source": "CST Anti-UAV train", "split": args.cst_split, "sequences": sorted({f["seq"] for f in frames}),
                 "frames": len(frames), "frames_with_box": sum(1 for f in frames if f["gt"]), "per_batch": args.cst_per_batch, "scale": 1.0}
    # one "epoch" = the stage 1 epoch length (1,075 steps of 8 images), whatever the mix
    steps_per_epoch = len(ds) // bs
    total = args.max_iters or args.epochs * steps_per_epoch

    model = build_model(mc)
    # anchor sizes are not parameters, so a checkpoint trained with other anchors loads unchanged
    model.load_state_dict(torch.load(ROOT / args.init, map_location="cpu"))
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=tc["momentum"], weight_decay=tc["weight_decay"])

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (0.001 + 0.999 * it / args.warmup)
        p = (it - args.warmup) / max(1, total - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    it, history, used_min, status = 0, [], 0.0, "running"
    state_p = run_dir / "state.pt"
    if state_p.exists() and persist:
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        model.load_state_dict(torch.load(run_dir / "last.pt", map_location="cpu"))
        model.to(device)
        opt.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["torch_rng"])
        it, history, used_min = st["iter"], st["history"], st["minutes_used"]
        print(f"resumed at iter {it}, {used_min:.1f} min used", flush=True)
    log_f = open(run_dir / "train_log.jsonl", "a")
    t_proc = time.time()

    def minutes():
        return used_min + (time.time() - t_proc) / 60

    def save_state(extra=None):
        torch.save(model.state_dict(), run_dir / "last.pt")
        torch.save({"iter": it, "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                    "history": history, "minutes_used": minutes(), **(extra or {})}, state_p)

    def write_result(final=False):
        write_json({"run": args.run, "status": status, "args": vars(args), "iterations": it, "planned_iterations": total,
                    "steps_per_epoch": steps_per_epoch, "minutes_used": minutes(), "time_limit_min": args.time_limit_min,
                    "history": history, "adaptation_data": adapt, "train_frames_antiuav": len(ds),
                    "final": final, "meta": run_metadata(cfg, {"stage": args.stage, "init": args.init, "anchor_sizes": mc["anchor_sizes"]})}, res_path)

    slow, run_loss, n_b, t0 = 0, 0.0, 0, time.time()
    while it < total:
        epoch, b = divmod(it, steps_per_epoch)
        if b == 0 or n_b == 0:
            perm = np.random.default_rng(cfg["seed"] * 1000 + epoch).permutation(len(ds))
            ds.epoch = epoch
            model.train()
            set_input_size(model, tc["crop"], tc["crop"])
        t_it = time.time()
        idx = perm[b * bs: b * bs + n_auv]
        batch = [ds[int(i)] for i in idx]
        if svan is not None:
            r = np.random.default_rng(cfg["seed"] * 7919 + it)
            batch += [svan.get(int(j), epoch, it) for j in r.integers(0, len(svan), args.svan_per_batch)]
        for gp in opt.param_groups:
            gp["lr"] = lr_at(it)
        opt.zero_grad(set_to_none=True)
        loss_val, n_img = 0.0, len(batch)
        for a in range(0, n_img, micro):
            mi = [x[0].to(device) for x in batch[a:a + micro]]
            mt = [{k: v.to(device) for k, v in x[1].items()} for x in batch[a:a + micro]]
            losses = model(mi, mt)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at iter {it}: {losses}")
            (loss * len(mi) / n_img).backward()
            loss_val += loss.item() * len(mi) / n_img
            del mi, mt, losses, loss
        opt.step()
        run_loss += loss_val
        n_b += 1
        it += 1
        slow = slow + 1 if time.time() - t_it > 6.0 else 0
        if it % 50 == 0:
            rec = {"iter": it, "epoch": epoch, "lr": lr_at(it), "loss": run_loss / n_b,
                   "s_per_iter": (time.time() - t0) / n_b, "minutes": round(minutes(), 1), **_mem_stats()}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(rec, flush=True)
        end_epoch = it % steps_per_epoch == 0 or it == total
        if end_epoch:
            _empty_cache(device)
            predict = make_predict_fn(model, device, mc, batch_size=mc.get("eval_batch_size", 1), load_image=load_image)
            val_preds = predict(val)
            _empty_cache(device)
            ep = (it - 1) // steps_per_epoch
            rec = {"epoch": ep, "iter": it, "train_loss": run_loss / max(n_b, 1), "val_ap50": ap_on(val, val_preds, cfg["eval"]["iou_thr"]),
                   "val_frames": len(val), "minutes": round(minutes(), 1)}
            history.append(rec)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(rec, flush=True)
            if persist:
                torch.save(model.state_dict(), run_dir / f"epoch{ep}.pt")
                write_json(val_preds, run_dir / f"val_preds_epoch{ep}.json")
            run_loss, n_b, t0 = 0.0, 0, time.time()
            model.train()
            set_input_size(model, tc["crop"], tc["crop"])
        if persist and (it % SAVE_EVERY == 0 or end_epoch):
            save_state()
            write_result()
        if minutes() > args.time_limit_min and it < total:
            status = f"stopped: wall-clock limit {args.time_limit_min} min exceeded at iter {it}/{total}"
            break
        if slow >= 200:
            status = f"stopped: 200 consecutive iterations slower than 6 s at iter {it}/{total}"
            break
    if status == "running":
        status = "complete"
    print("STATUS", status, flush=True)
    if persist:
        save_state()
        write_result(final=True)
    log_f.close()


if __name__ == "__main__":
    main()
