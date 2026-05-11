from __future__ import annotations

from typing import Any

import torch


import torch
from typing import Any, Dict

def relabel_cls_by_box_area(
    batch: Dict[str, Any],
    small_area: float = 100.0,
    medium_area: float = 900.0,
) -> Dict[str, Any]:
    """按 bbox 面积重新划分类别"""

    if "img" not in batch or "bboxes" not in batch or "cls" not in batch:
        return batch

    bboxes = batch["bboxes"]
    cls = batch["cls"]

    if not isinstance(bboxes, torch.Tensor) or not isinstance(cls, torch.Tensor) or bboxes.numel() == 0:
        return batch

    # 图像尺寸
    h, w = batch["img"].shape[2:]

    # 计算面积（像素）
    areas = (bboxes[:, 2] * w) * (bboxes[:, 3] * h)

    cls = cls.squeeze(-1)
    new_cls = cls.clone()

    # ===================== 类别 0 =====================
    mask0 = cls == 0
    new_cls[mask0] = 6                          # 默认大
    new_cls[mask0 & (areas < medium_area)] = 5  # 中
    new_cls[mask0 & (areas < small_area)] = 0   # 小

    # ===================== 类别 1 =====================
    # 不变（不用动）

    # ===================== 类别 2 =====================
    mask2 = cls == 2
    new_cls[mask2] = 8                          # 默认大
    new_cls[mask2 & (areas < medium_area)] = 7  # 中
    new_cls[mask2 & (areas < small_area)] = 2   # 小

    # ===================== 类别 3、4 =====================
    # 不变（不用动）

    batch["cls"] = new_cls.unsqueeze(-1)
    return batch