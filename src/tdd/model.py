"""Detector: torchvision Faster R-CNN ResNet50-FPN (BSD-3-Clause), COCO-pretrained.

Why this model (see README "Model choice"): permissive license (Ultralytics YOLO is
AGPL-3.0), FPN starts at stride 4 (P2) which suits small targets, runs on Apple MPS
with stock PyTorch ops.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(mc: dict) -> torch.nn.Module:
    assert mc["arch"] == "fasterrcnn_resnet50_fpn"
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if mc["weights"] == "DEFAULT" else None
    m = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=mc["trainable_backbone_layers"],
        min_size=mc["eval_min_size"], max_size=mc["eval_max_size"],
        box_score_thresh=mc["box_score_thresh"],
        box_detections_per_img=mc["detections_per_img"],
    )
    # Smaller anchors (one size per FPN level); still 3 aspect ratios per location,
    # so the pretrained RPN head weights keep their shape.
    sizes = tuple((s,) for s in mc["anchor_sizes"])
    ratios = (tuple(mc["aspect_ratios"]),) * len(sizes)
    m.rpn.anchor_generator = AnchorGenerator(sizes, ratios)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)  # background + drone
    return m


def set_input_size(m, min_size: int, max_size: int) -> None:
    """GeneralizedRCNNTransform resizes so the short side == min_size; we always feed
    images at the intended scale, so set it to match and avoid hidden resampling."""
    m.transform.min_size = (min_size,)
    m.transform.max_size = max_size


def to_tensor(gray: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(gray).to(device=device, dtype=torch.float32).div_(255.0)
    return t.unsqueeze(0).expand(3, -1, -1)


def make_predict_fn(m, device, mc: dict, batch_size: int = 4, load_image=None):
    m.eval()

    @torch.no_grad()
    def predict(frames, scale: float = 1.0):
        out = []
        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            imgs = []
            for fr in chunk:
                g = load_image(fr["image"])
                if scale != 1.0:
                    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                imgs.append(g)
            h, w = imgs[0].shape
            set_input_size(m, min(h, w), max(h, w))
            res = m([to_tensor(g, device) for g in imgs])
            for r in res:
                b = r["boxes"].float().cpu().numpy() / scale
                s = r["scores"].float().cpu().numpy()[:, None]
                out.append(np.concatenate([b, s], 1).tolist())
        return out

    return predict
