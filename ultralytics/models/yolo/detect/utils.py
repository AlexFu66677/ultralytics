from __future__ import annotations

from typing import Any

import torch


def relabel_cls_by_box_area(
    batch: dict[str, Any], small_area: float = 100.0, medium_area: float = 900.0
) -> dict[str, Any]:
    """Relabel classes by bbox area measured on the current input image size."""

    if "img" not in batch or "bboxes" not in batch or "cls" not in batch:
        return batch

    bboxes = batch["bboxes"]
    cls = batch["cls"]

    if not isinstance(bboxes, torch.Tensor) or not isinstance(cls, torch.Tensor) or bboxes.numel() == 0:
        return batch

    h, w = batch["img"].shape[2:]

    areas = (bboxes[:, 2] * w) * (bboxes[:, 3] * h)

    # 变成一维避免 broadcast
    cls = cls.squeeze(-1)

    new_cls = cls.clone()

    mask0 = cls == 0
    new_cls[mask0] = 2
    new_cls[mask0 & (areas < 900)] = 1
    new_cls[mask0 & (areas < 100)] = 0

    mask2 = cls == 1
    new_cls[mask2] = 4
    new_cls[mask2 & (areas < 200)] = 3

    batch["cls"] = new_cls.unsqueeze(-1)
    return batch