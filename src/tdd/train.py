"""Fine-tune Faster R-CNN on Anti-UAV300 IR (train split), select epoch on val AP@0.5.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.train [--config configs/default.yaml]

Writes runs/frcnn/{last.pt,best.pt,train_log.jsonl} (not committed) and
results/model/train_log.json (committed).
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
from torch.utils.data import DataLoader, Dataset

from .common import ROOT, load_config, path, read_json, run_metadata, set_seed, write_json
from .evaluate import ap_on, load_image
from .model import build_model, make_predict_fn, set_input_size


class CropDataset(Dataset):
    """Random native-resolution crops (optionally after down-scaling) of train frames."""

    def __init__(self, frames, tc: dict, seed: int):
        self.frames = frames if tc["include_empty_frames"] else [f for f in frames if f["gt"]]
        self.tc = tc
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        rng = random.Random(hash((self.seed, self.epoch, idx)) & 0xFFFFFFFF)
        tc = self.tc
        fr = self.frames[idx]
        g = load_image(fr["image"])
        boxes = np.asarray(fr["gt"], np.float32).reshape(-1, 4)
        sj = tc.get("scale_jitter")
        if sj and rng.random() < sj["p"]:
            s = rng.uniform(sj["min"], sj["max"])
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            boxes = boxes * s
        H, W = g.shape
        c = tc["crop"]
        if H < c or W < c:  # pad small images
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
            keep = (area1 >= 0.5 * area0) & (b[:, 2] - b[:, 0] >= 2) & (b[:, 3] - b[:, 1] >= 2)
            boxes = b[keep]
        if rng.random() < tc["hflip_p"]:
            g = g[:, ::-1]
            if len(boxes):
                boxes = np.stack([c - boxes[:, 2], boxes[:, 1], c - boxes[:, 0], boxes[:, 3]], 1)
        img = torch.from_numpy(np.ascontiguousarray(g)).float().div_(255.0).unsqueeze(0).expand(3, -1, -1).contiguous()
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return img, target


def _empty_cache(device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def _mem_stats() -> dict:
    out = {}
    if torch.backends.mps.is_available():
        out["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 3)
        out["mps_current_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 3)
    return out


def collate(batch):
    return tuple(zip(*batch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="frcnn")
    ap.add_argument("--max-iters", type=int, default=0, help="smoke test only")
    ap.add_argument("--results-dir", default="results/model")
    ap.add_argument("--resume", action="store_true",
                    help="continue from runs/<run>/train_state.pt (end of the last finished epoch) if it exists")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    tc, mc = cfg["train"], cfg["model"]
    device = torch.device(tc["device"] if torch.backends.mps.is_available() else "cpu")
    run_dir = ROOT / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    man = path(cfg["data"]["manifests_dir"])
    train = read_json(man / "antiuav_train.json")
    val = read_json(man / "antiuav_val.json")[:: max(1, tc["val_stride_during_training"] // cfg["split"]["stride"]["val"])]
    if args.max_iters:  # smoke test: tiny val so the check takes seconds
        val = val[:16]
    ds = CropDataset(train, tc, cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])
    micro = int(tc.get("micro_batch_size", tc["batch_size"]))
    assert tc["batch_size"] % micro == 0, "batch_size must be a multiple of micro_batch_size"
    dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, num_workers=tc["num_workers"],
                    collate_fn=collate, generator=g, persistent_workers=False, drop_last=True)

    model = build_model(mc).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=tc["lr"], momentum=tc["momentum"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * len(dl)
    if args.max_iters:
        total = args.max_iters

    def lr_at(it):
        if it < tc["warmup_iters"]:
            return tc["lr"] * (0.001 + 0.999 * it / tc["warmup_iters"])
        p = (it - tc["warmup_iters"]) / max(1, total - tc["warmup_iters"])
        return tc["lr"] * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    history, best_ap, it, start_epoch, resumed = [], -1.0, 0, 0, None
    state_p = run_dir / "train_state.pt"
    if args.resume and state_p.exists():
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        if st["total_iters"] != total:
            raise RuntimeError(f"schedule mismatch: saved total {st['total_iters']} != {total}")
        model.load_state_dict(torch.load(run_dir / f"epoch{st['epoch']}.pt", map_location="cpu"))
        opt.load_state_dict(st["optimizer"])
        it, start_epoch = st["iter"], st["epoch"] + 1
        if "torch_rng" in st:
            torch.set_rng_state(st["torch_rng"])
        prev = read_json(ROOT / args.results_dir / "train_log.json")
        history = [h for h in prev["history"] if h["epoch"] <= st["epoch"]]
        best_ap = max(h["val_ap50_subset"] for h in history)
        # replay the generator draws of the finished epochs (worker base seed at
        # iterator creation + the shuffle permutation at the first batch), so the
        # remaining epochs see the same data order as an uninterrupted run
        for _ in range(start_epoch):
            _it = iter(dl)
            next(_it)
            del _it
        resumed = {"from_epoch": st["epoch"], "iter": it, "rng_state_restored": "torch_rng" in st,
                   "at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        print("resumed", resumed, "history", history, flush=True)
    log_f = open(run_dir / "train_log.jsonl", "a" if resumed else "w")
    if resumed:
        log_f.write(json.dumps({"resumed": resumed}) + "\n")
    t_start = time.time()
    for epoch in range(start_epoch, tc["epochs"]):
        ds.epoch = epoch
        model.train()
        set_input_size(model, tc["crop"], tc["crop"])
        run_loss, n_b, t0 = 0.0, 0, time.time()
        for imgs, tgts in dl:
            for gp in opt.param_groups:
                gp["lr"] = lr_at(it)
            # The loader still yields the full batch (same shuffle and RNG draws as
            # before); it is split into micro-batches whose gradients are accumulated,
            # so one optimizer step still sees batch_size images. Faster R-CNN losses
            # are per-image-balanced means, so mean-of-micro-means == full-batch mean.
            opt.zero_grad(set_to_none=True)
            loss_val, loss_parts = 0.0, {}
            n_img = len(imgs)
            for a in range(0, n_img, micro):
                mi = [i.to(device) for i in imgs[a:a + micro]]
                mt = [{k: v.to(device) for k, v in t.items()} for t in tgts[a:a + micro]]
                losses = model(mi, mt)
                w = len(mi) / n_img
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at iter {it}: {losses}")
                (loss * w).backward()
                loss_val += loss.item() * w
                for k, v in losses.items():
                    loss_parts[k] = loss_parts.get(k, 0.0) + v.item() * w
                del mi, mt, losses, loss
            opt.step()
            run_loss += loss_val
            n_b += 1
            it += 1
            if it % 50 == 0:
                rec = {"iter": it, "epoch": epoch, "lr": lr_at(it), "loss": run_loss / n_b,
                       **{k: round(v, 4) for k, v in loss_parts.items()},
                       "s_per_iter": (time.time() - t0) / n_b, **_mem_stats()}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(rec, flush=True)
            if args.max_iters and it >= args.max_iters:
                break
        # --- per-epoch val AP@0.5 (checkpoint selection only) ---
        _empty_cache(device)
        predict = make_predict_fn(model, device, mc, batch_size=mc.get("eval_batch_size", 4), load_image=load_image)
        val_preds = predict(val)
        _empty_cache(device)
        val_ap = ap_on(val, val_preds, cfg["eval"]["iou_thr"])
        write_json(val_preds, run_dir / f"val_preds_epoch{epoch}.json")
        rec = {"epoch": epoch, "train_loss": run_loss / max(n_b, 1), "val_ap50_subset": val_ap,
               "val_frames": len(val), "epoch_minutes": (time.time() - t0) / 60}
        history.append(rec)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(rec, flush=True)
        torch.save(model.state_dict(), run_dir / "last.pt")
        torch.save(model.state_dict(), run_dir / f"epoch{epoch}.pt")
        # full training state, so an interrupted run can be resumed on the same schedule
        torch.save({"epoch": epoch, "iter": it, "optimizer": opt.state_dict(), "total_iters": total,
                    "torch_rng": torch.get_rng_state()}, run_dir / "train_state.pt")
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), run_dir / "best.pt")
            if len(val) == len(read_json(man / "antiuav_val.json")):
                # val preds of the selected checkpoint = the evaluation cache for the full val split
                write_json(val_preds, run_dir / "preds_antiuav_val.json")
        write_json({"history": history, "best_epoch_so_far": max(history, key=lambda r: r["val_ap50_subset"])["epoch"],
                    "iterations_so_far": it, "planned_iterations": total, "complete": False,
                    "resumed": resumed, "meta": run_metadata(cfg, {"run": args.run})},
                   ROOT / args.results_dir / "train_log.json")
        if args.max_iters and it >= args.max_iters:
            break
    log_f.close()
    best_epoch = max(history, key=lambda r: r["val_ap50_subset"])["epoch"]
    write_json({"history": history, "best_epoch": best_epoch, "iterations": it, "complete": True,
                "resumed": resumed,
                "val_frames_for_selection": len(val),
                "train_frames": len(ds), "minutes_this_process": (time.time() - t_start) / 60,
                "device": str(device), "meta": run_metadata(cfg, {"run": args.run})},
               ROOT / args.results_dir / ("train_log.json" if not args.max_iters else "smoke_train_log.json"))


if __name__ == "__main__":
    main()
