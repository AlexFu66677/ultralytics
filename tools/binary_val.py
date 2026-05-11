from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import HELP_URL, load_dataset_cache_file
from ultralytics.utils import LOGGER, TQDM, colorstr, ops
from ultralytics.utils.metrics import Metric, ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.metrics import ConfusionMatrix


CLASS_GROUPS = {0: [0], 1: [1]}
NAMES = {0: "class0", 1: "class1"}


class FolderDetectionDataset(YOLODataset):
    """YOLO-format dataset loaded directly from image and label folders."""

    def __init__(self, image_dir: str, label_dir: str, *args, **kwargs) -> None:
        self.image_dir = Path(image_dir).resolve()
        self.label_dir = Path(label_dir).resolve()
        super().__init__(img_path=str(self.image_dir), *args, **kwargs)

    def get_labels(self) -> List[Dict[str, Any]]:
        self.label_files = []
        for im_file in self.im_files:
            relative = Path(im_file).resolve().relative_to(self.image_dir)
            self.label_files.append(str((self.label_dir / relative).with_suffix(".txt")))

        cache_path = self.label_dir.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            from ultralytics.data.dataset import DATASET_CACHE_VERSION
            from ultralytics.data.utils import get_hash

            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except Exception:
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        for key in ("hash", "version", "msgs"):
            cache.pop(key, None)
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]
        return labels


def resolve_data_dirs(root: str) -> tuple[Path, Path]:
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(root)

    image_candidates = [root_path / "images", root_path / "imgs", root_path]
    label_candidates = [root_path / "labels", root_path / "label", root_path / "annotations", root_path]

    image_dir = next((p for p in image_candidates if p.exists() and p.is_dir()), None)
    label_dir = next((p for p in label_candidates if p.exists() and p.is_dir()), None)
    if image_dir is None or label_dir is None:
        raise FileNotFoundError(f"Could not find image/label folders under {root_path}")
    return image_dir, label_dir


def build_dataset_args(imgsz: int, rect: bool, cache: bool | str = False):
    args = get_cfg(
        overrides={
            "imgsz": imgsz,
            "rect": rect,
            "cache": cache,
            "single_cls": False,
            "task": "detect",
            "classes": None,
            "fraction": 1.0,
            "mode": "val",
        }
    )
    return args


def build_dataset(data_dir: str, imgsz: int, batch: int, rect: bool = True):
    image_dir, label_dir = resolve_data_dirs(data_dir)
    data = {"names": NAMES, "channels": 3, "nc": len(NAMES)}
    dataset_args = build_dataset_args(imgsz=imgsz, rect=rect)
    dataset = FolderDetectionDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        imgsz=dataset_args.imgsz,
        batch_size=batch,
        augment=False,
        hyp=dataset_args,
        rect=dataset_args.rect,
        cache=dataset_args.cache or None,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix=colorstr("val: "),
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
    )
    dataloader = build_dataloader(dataset, batch, workers=0, shuffle=False, rank=-1, drop_last=False, pin_memory=False)
    return dataset, dataloader


def map_logits_to_binary(preds: torch.Tensor, model_nc: int, class_groups: Dict[int, List[int]]) -> torch.Tensor:
    """Convert model class logits into 2 mapped classes before NMS."""
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    if preds.shape[-1] == 6:
        return preds

    bs, _, num_boxes = preds.shape
    extra = preds.shape[1] - 4 - model_nc
    mapped = preds.new_zeros((bs, 4 + len(class_groups) + extra, num_boxes))
    mapped[:, :4, :] = preds[:, :4, :]
    if extra > 0:
        mapped[:, 4 + len(class_groups) :, :] = preds[:, 4 + model_nc :, :]

    for target_cls, src_classes in class_groups.items():
        valid_classes = [c for c in src_classes if c < model_nc]
        if not valid_classes:
            continue
        scores = preds[:, 4 + torch.tensor(valid_classes, device=preds.device), :]
        mapped[:, 4 + target_cls, :] = scores.amax(dim=1)
    return mapped


def postprocess_predictions(
    preds: torch.Tensor, conf: float, iou: float, max_det: int, model_nc: int, class_groups: Dict[int, List[int]]
) -> List[Dict[str, torch.Tensor]]:
    """Run mapped-class NMS and return prediction dicts compatible with Ultralytics metrics helpers."""
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    if preds.shape[-1] == 6:
        outputs = []
        for pred in preds:
            pred = pred.clone()
            mapped_cls = torch.full_like(pred[:, 5], -1)
            for dst, src_list in class_groups.items():
                for src in src_list:
                    mapped_cls[pred[:, 5] == src] = dst
            keep = (mapped_cls >= 0) & (pred[:, 4] >= conf)
            pred = pred[keep]
            mapped_cls = mapped_cls[keep]
            if pred.numel():
                pred[:, 5] = mapped_cls
            outputs.append(pred[:max_det])
    else:
        mapped_preds = map_logits_to_binary(preds, model_nc=model_nc, class_groups=class_groups)
        outputs = non_max_suppression(
            mapped_preds,
            conf_thres=conf,
            iou_thres=iou,
            nc=len(class_groups),
            multi_label=False,
            agnostic=False,
            max_det=max_det,
        )

    return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]


def prepare_batch(sample_index: int, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    idx = batch["batch_idx"] == sample_index
    cls = batch["cls"][idx].squeeze(-1)
    bbox = batch["bboxes"][idx]
    ori_shape = batch["ori_shape"][sample_index]
    imgsz = batch["img"].shape[2:]
    ratio_pad = batch["ratio_pad"][sample_index]
    if cls.shape[0]:
        bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=device)[[1, 0, 1, 0]]
    return {
        "cls": cls,
        "bboxes": bbox,
        "ori_shape": ori_shape,
        "imgsz": imgsz,
        "ratio_pad": ratio_pad,
        "im_file": batch["im_file"][sample_index],
    }


def match_predictions(pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, iouv: torch.Tensor):
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    iou = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def process_batch(pred: Dict[str, torch.Tensor], target: Dict[str, Any], iouv: torch.Tensor) -> np.ndarray:
    if target["cls"].shape[0] == 0 or pred["cls"].shape[0] == 0:
        return np.zeros((pred["cls"].shape[0], iouv.shape[0]), dtype=bool)
    iou = box_iou(target["bboxes"], pred["bboxes"])
    return match_predictions(pred["cls"], target["cls"], iou, iouv)


def validate_mapped_detection(
    model_path: str,
    data_dir: str,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.7,
    device: str = "",
    max_det: int = 300,
    project: str | None = None,
    name: str | None = None,
    plots: bool = False,
):
    confmat = ConfusionMatrix(names=NAMES, task="detect")
    save_cfg = get_cfg(overrides={"project": project, "name": name, "mode": "val", "task": "detect"})
    save_dir = get_save_dir(save_cfg)
    save_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(model_path)
    model = yolo.model
    dev = select_device(device)
    model = model.to(dev).eval()
    model.half() if dev.type != "cpu" else model.float()
    half = dev.type != "cpu"
    model_nc = len(model.names)
    dataset, dataloader = build_dataset(data_dir=data_dir, imgsz=imgsz, batch=batch)
    iouv = torch.linspace(0.5, 0.95, 10, device=dev)

    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    seen = 0

    for batch_data in TQDM(dataloader, desc="Mapped val", total=len(dataloader)):
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(dev, non_blocking=dev.type == "cuda")
        batch_data["img"] = (batch_data["img"].half() if half else batch_data["img"].float()) / 255.0

        with torch.inference_mode():
            preds = model(batch_data["img"])
        preds = postprocess_predictions(preds, conf=conf, iou=iou, max_det=max_det, model_nc=model_nc, class_groups=CLASS_GROUPS)

        for si, pred in enumerate(preds):
            seen += 1
            target = prepare_batch(si, batch_data, dev)
            if target["cls"].numel():
                valid = target["cls"] < len(NAMES)
                target["cls"] = target["cls"][valid]
                target["bboxes"] = target["bboxes"][valid]

            confmat.process_batch(pred, target)
            no_pred = pred["cls"].shape[0] == 0
            stats["tp"].append(process_batch(pred, target, iouv))
            stats["conf"].append(np.zeros(0) if no_pred else pred["conf"].detach().cpu().numpy())
            stats["pred_cls"].append(np.zeros(0) if no_pred else pred["cls"].detach().cpu().numpy())
            stats["target_cls"].append(target["cls"].detach().cpu().numpy())

    merged = {k: np.concatenate(v, 0) if len(v) else np.zeros(0) for k, v in stats.items()}
    if merged["target_cls"].size == 0:
        raise RuntimeError("No valid labels found after filtering to classes 0 and 1.")

    metric = Metric()
    results = ap_per_class(
        merged["tp"],
        merged["conf"],
        merged["pred_cls"],
        merged["target_cls"],
        plot=plots,
        save_dir=save_dir,
        names=NAMES,
        prefix="MappedBox",
    )[2:]
    metric.nc = len(NAMES)
    metric.update(results)

    LOGGER.info("Validation images: %d", seen)
    LOGGER.info(
        "mapped metrics: P=%.4f, R=%.4f, mAP50=%.4f, mAP50-95=%.4f",
        metric.mp,
        metric.mr,
        metric.map50,
        metric.map,
    )
    for i, cls_idx in enumerate(metric.ap_class_index):
        LOGGER.info(
            "class %s: P=%.4f, R=%.4f, mAP50=%.4f, mAP50-95=%.4f",
            NAMES[int(cls_idx)],
            metric.p[i],
            metric.r[i],
            metric.ap50[i],
            metric.ap[i],
        )
    confmat.plot(normalize = True,save_dir=save_dir)
    confmat.plot(normalize = False,save_dir=save_dir)
    return {
        "precision": metric.mp,
        "recall": metric.mr,
        "map50": metric.map50,
        "map50_95": metric.map,
        "per_class": {
            NAMES[int(cls_idx)]: {
                "precision": float(metric.p[i]),
                "recall": float(metric.r[i]),
                "map50": float(metric.ap50[i]),
                "map50_95": float(metric.ap[i]),
            }
            for i, cls_idx in enumerate(metric.ap_class_index)
        },
        "save_dir": str(save_dir),
        "image_dir": str(dataset.image_dir),
        "label_dir": str(dataset.label_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate detection with mapped classes from a folder of images/labels.")
    parser.add_argument("--model", required=True, help="Model path, e.g. runs/detect/train/weights/best.pt")
    parser.add_argument("--data-dir", required=True, help="Dataset root folder containing images and labels")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="Validation device")
    parser.add_argument("--max-det", dest="max_det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--project", default="runs/mapped_val", help="Project directory")
    parser.add_argument("--name", default="exp", help="Run name")
    parser.add_argument("--plots", action="store_true", help="Save PR/F1/P/R curves")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_mapped_detection(
        model_path=args.model,
        data_dir=args.data_dir,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
