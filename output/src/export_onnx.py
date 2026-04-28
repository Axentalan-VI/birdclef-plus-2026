"""Export a trained timm-CNN checkpoint to ONNX + INT8 dynamic quantization."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch

from .config import ASSETS_DIR, N_MELS, SPEC_FRAMES
from .models import build_model
from .taxonomy import num_classes


def export_cnn(ckpt_path: Path, out_path: Path, backbone: str, opset: int = 17) -> Path:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model("timm", backbone=backbone, num_classes=num_classes(), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 1, N_MELS, SPEC_FRAMES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path.as_posix(),
        input_names=["mel"], output_names=["logits"],
        dynamic_axes={"mel": {0: "B"}, "logits": {0: "B"}},
        opset_version=opset,
    )
    onnx.checker.check_model(str(out_path))
    return out_path


def export_embhead(ckpt_path: Path, out_path: Path, opset: int = 17) -> Path:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    emb_dim = int(ckpt.get("emb_dim", 1280))
    hidden = int(ckpt.get("cfg", {}).get("hidden", 512))
    drop = float(ckpt.get("cfg", {}).get("drop", 0.3))
    model = build_model("embhead", emb_dim=emb_dim, num_classes=num_classes(), hidden=hidden, drop=drop)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, emb_dim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path.as_posix(),
        input_names=["emb"], output_names=["logits"],
        dynamic_axes={"emb": {0: "B"}, "logits": {0: "B"}},
        opset_version=opset,
    )
    onnx.checker.check_model(str(out_path))
    return out_path


# Backward-compat alias.
export = export_cnn


def quantize_int8(fp32_path: Path, int8_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(
        model_input=fp32_path.as_posix(),
        model_output=int8_path.as_posix(),
        weight_type=QuantType.QInt8,
    )
    return int8_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--kind", choices=["cnn", "embhead"], default="cnn")
    ap.add_argument("--backbone", default="tf_efficientnet_b0.ns_jft_in1k",
                    help="only used when --kind cnn")
    ap.add_argument("--name", required=True, help="basename for the exported ONNX files")
    args = ap.parse_args()

    fp32 = ASSETS_DIR / f"{args.name}.onnx"
    int8 = ASSETS_DIR / f"{args.name}.int8.onnx"

    if args.kind == "cnn":
        export_cnn(args.ckpt, fp32, backbone=args.backbone)
    else:
        export_embhead(args.ckpt, fp32)
    print(f"fp32 -> {fp32}  ({fp32.stat().st_size/1e6:.1f} MB)")
    quantize_int8(fp32, int8)
    print(f"int8 -> {int8}  ({int8.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
