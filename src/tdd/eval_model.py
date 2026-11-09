"""Evaluate the fine-tuned detector with the same protocol as the baseline.

Usage: PYTHONPATH=src .venv/bin/python -m tdd.eval_model --run frcnn [--ckpt best.pt]
"""
from __future__ import annotations

import argparse
import hashlib

import torch

from .common import ROOT, load_config, set_seed
from .evaluate import evaluate_all, load_image
from .model import build_model, make_predict_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="frcnn")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", default="results/model")
    ap.add_argument("--no-reuse", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device(cfg["train"]["device"] if torch.backends.mps.is_available() else "cpu")
    ckpt = ROOT / "runs" / args.run / args.ckpt
    # Built exactly as in training (COCO init -> FrozenBatchNorm backbone), then
    # every parameter is overwritten by the fine-tuned checkpoint.
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    predict = make_predict_fn(model, device, cfg["model"], batch_size=cfg["model"].get("eval_batch_size", 4),
                              load_image=load_image)
    evaluate_all(cfg, predict, name=args.run, out_dir=ROOT / args.out, reuse=not args.no_reuse,
                 method_info={"method": "Faster R-CNN R50-FPN (torchvision, COCO-pretrained) fine-tuned on Anti-UAV300 IR train",
                              "checkpoint": f"runs/{args.run}/{args.ckpt}", "checkpoint_sha256": sha,
                              "model_cfg": cfg["model"], "train_cfg": cfg["train"]})


if __name__ == "__main__":
    main()
