#!/usr/bin/env python3
"""
Measure ResNeXt101_64x4d inference latency & memory vs batch size, then plot.

Jargon:
- "Amortized latency": elapsed time per image, i.e., (batch inference time / batch_size).
- "Max GPU memory allocated": peak memory allocated by PyTorch on the selected CUDA device
  during the measurement window. We reset caches between batches to keep measurements comparable.

This script:
  1) Scans an image directory with files named "[class_idx]_[uid].png"
  2) For each batch size:
      - Runs a fixed number of batches (default 5), or until the dataset is exhausted
      - Measures average amortized latency (ms / image)
      - Captures maximum GPU memory allocated (MB)
  3) Saves results to CSV and creates a dual-axis plot: latency (ms) and memory (% of device)
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

from PIL import Image, ImageFile, PngImagePlugin  # type: ignore

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# --------------------------------------------------------------------------------------
# Pillow safety tuning (robust but conservative)
# --------------------------------------------------------------------------------------

# Allow loading slightly malformed/truncated PNGs instead of crashing the benchmark.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Keep Pillow's decompression-bomb protections, but avoid spurious failures on legitimate
# files with moderately large metadata. These are modest caps, not "unlimited".
# (We still implement a chunk-stripping fallback below for truly bloated files.)
if hasattr(PngImagePlugin, "MAX_TEXT_CHUNK"):
    PngImagePlugin.MAX_TEXT_CHUNK = max(getattr(PngImagePlugin, "MAX_TEXT_CHUNK", 1_000_000), 4 * 1024 * 1024)
if hasattr(PngImagePlugin, "MAX_TEXT_MEMORY"):
    PngImagePlugin.MAX_TEXT_MEMORY = max(getattr(PngImagePlugin, "MAX_TEXT_MEMORY", 64 * 1024 * 1024), 64 * 1024 * 1024)


# --------------------------------------------------------------------------------------
# PNG metadata stripping fallback (handles "Decompressed Data Too Large")
# --------------------------------------------------------------------------------------

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Chunk types to drop: oversized text/ICC/XMP payloads that are irrelevant to inference.
_DROP_CHUNK_TYPES = {b"iCCP", b"iTXt", b"zTXt", b"tEXt", b"eXIf"}

def _strip_png_text_chunks(raw: bytes) -> bytes:
    """
    Remove ancillary PNG chunks that often carry huge compressed metadata (iCCP/iTXt/zTXt/tEXt/eXIf).
    The function preserves all other chunks (IHDR, IDAT, PLTE, IEND, etc.) and their CRCs.

    Args:
        raw: Original PNG file bytes.

    Returns:
        Cleaned PNG bytes with bloated text/ICC chunks removed. If the input is not a PNG,
        the original bytes are returned unchanged.
    """
    if not raw.startswith(PNG_SIGNATURE):
        return raw

    # Output buffer with PNG signature
    out = bytearray(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    n = len(raw)

    # Parse chunk-by-chunk: [length (4)][type (4)][data (length)][crc (4)]
    while offset + 8 <= n:
        # Read length and type
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        ctype = raw[offset + 4:offset + 8]
        offset += 8

        # Read data and CRC
        if offset + length + 4 > n:
            # Malformed; bail out with what we have (let Pillow handle the rest)
            return bytes(raw)

        data = raw[offset:offset + length]
        crc = raw[offset + length:offset + length + 4]
        offset += length + 4

        # Drop selected text/ICC/XMP chunks; keep others verbatim
        if ctype in _DROP_CHUNK_TYPES:
            # Skip writing this chunk
            continue

        # Write the untouched chunk back out
        out += struct.pack(">I", length)
        out += ctype
        out += data
        out += crc

        if ctype == b"IEND":
            break

    return bytes(out)


def _safe_open_png(path: str) -> Image.Image:
    """
    Robust image opener that:
      1) Tries a normal Pillow open.
      2) On PNG metadata explosion errors, re-reads the file, strips bloated metadata chunks
         (iCCP/iTXt/zTXt/tEXt/eXIf) in-memory, then re-opens from cleaned bytes.
      3) Always returns an RGB image or raises a terminal exception.

    This ensures we do not skip files; instead we sanitize and proceed.
    """
    try:
        # Primary attempt
        with Image.open(path) as im:
            return im.convert("RGB")
    except ValueError as e:
        # Specific Pillow guard: oversized decompressed text/ICC data
        if "Decompressed Data Too Large" in str(e) or "Exceeded limit" in str(e):
            with open(path, "rb") as f:
                raw = f.read()
            cleaned = _strip_png_text_chunks(raw)
            with Image.open(io.BytesIO(cleaned)) as im2:
                return im2.convert("RGB")
        # Re-raise for unrelated ValueErrors
        raise
    except OSError:
        # As a last-ditch attempt, try strip+open for possibly odd PNGs mislabeled as errors.
        with open(path, "rb") as f:
            raw = f.read()
        cleaned = _strip_png_text_chunks(raw)
        with Image.open(io.BytesIO(cleaned)) as im2:
            return im2.convert("RGB")


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

class ImageDataset(Dataset):
    """Tiny dataset for flat folders of PNGs named `[class_idx]_[uid].png`."""

    def __init__(self, image_dir: str, transform: Optional[T.Compose] = None) -> None:
        self.image_dir: str = image_dir
        self.image_files: List[str] = [
            f for f in os.listdir(image_dir) if f.lower().endswith(".png")
        ]
        self.transform = transform

        if not self.image_files:
            raise FileNotFoundError(
                f"No .png files found in directory: {image_dir!r}. "
                "Expected filenames like '123_000001.png'."
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Tensor | int]:
        image_path = os.path.join(self.image_dir, self.image_files[idx])

        # Robust open that strips oversized metadata if needed; always returns RGB.
        image = _safe_open_png(image_path)

        # Parse label from filename prefix before underscore
        name = os.path.basename(image_path)
        try:
            label = int(name.split("_")[0])
        except Exception:
            label = 0  # fallback

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}


def build_dataset(image_dir: str) -> Dataset:
    """Construct transforms and dataset for ResNeXt101."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return ImageDataset(image_dir=image_dir, transform=transform)


# --------------------------------------------------------------------------------------
# Measurement core
# --------------------------------------------------------------------------------------

@dataclass
class MeasureConfig:
    batch_sizes: List[int]
    max_batches: int
    num_workers: int
    device: torch.device
    pin_memory: bool
    autocast: bool
    dtype: Optional[torch.dtype]  # if autocast==True, dtype is the compute type for autocast


@dataclass
class MeasureResult:
    batch_size: int
    amortized_latency_ms: Optional[float]  # average per-image latency (ms)
    max_memory_mb: Optional[float]         # peak memory during measurement (MB)
    status: str                            # "ok", "oom", or "error"
    note: str                              # additional info (exception text, etc.)


def _inference_timing_gpu(model: torch.nn.Module, batch: Tensor, device: torch.device,
                          autocast: bool, dtype: Optional[torch.dtype]) -> float:
    """
    Returns elapsed milliseconds for one forward pass using CUDA events.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        if autocast:
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = model(batch)
        else:
            _ = model(batch)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))  # milliseconds


def _inference_timing_cpu(model: torch.nn.Module, batch: Tensor,
                          autocast: bool, dtype: Optional[torch.dtype]) -> float:
    """
    Returns elapsed milliseconds for one forward pass on CPU using perf_counter.
    (autocast path included for API symmetry; it has no effect on CPU here)
    """
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(batch)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def measure_for_batch_size(
    *,
    ds: Dataset,
    batch_size: int,
    cfg: MeasureConfig,
) -> MeasureResult:
    """
    Run up to cfg.max_batches, measure (1) average per-image latency and (2) max GPU memory.
    If OOM occurs, report status and stop early for this batch size.
    """
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,  # shuffle each traversal for representativeness
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=False,
    )

    # Build model once per batch-size run to capture realistic memory behavior
    try:
        model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT).to(cfg.device)
        model.eval()
    except RuntimeError as e:
        return MeasureResult(
            batch_size=batch_size,
            amortized_latency_ms=None,
            max_memory_mb=None,
            status="error",
            note=f"Model init failed: {e}",
        )

    per_batch_ms: List[float] = []
    observed_max_mem: float = 0.0

    # Clear any prior CUDA stats
    if cfg.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(cfg.device)
        torch.cuda.empty_cache()

    seen_batches = 0
    status = "ok"
    note = ""

    try:
        for i, batch in enumerate(loader):
            if seen_batches >= cfg.max_batches:
                break

            images = batch["image"].to(cfg.device, non_blocking=True)

            try:
                if cfg.device.type == "cuda":
                    elapsed_ms = _inference_timing_gpu(model, images, cfg.device, cfg.autocast, cfg.dtype)
                    # Track peak after each forward
                    peak = torch.cuda.max_memory_allocated(cfg.device)
                    observed_max_mem = max(observed_max_mem, float(peak))
                else:
                    elapsed_ms = _inference_timing_cpu(model, images, cfg.autocast, cfg.dtype)

                # Amortize per image
                per_img_ms = elapsed_ms / max(1, images.shape[0])
                per_batch_ms.append(per_img_ms)

            except RuntimeError as e:
                # Catch OOMs or other runtime errors; mark and stop trying larger loads for this size
                status = "oom" if "out of memory" in str(e).lower() else "error"
                note = f"RuntimeError on batch {i}: {e}"
                break
            finally:
                # Free per-iteration tensors aggressively
                del images
                if cfg.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            seen_batches += 1

    finally:
        del model
        if cfg.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Summarize
    amortized_latency_ms = float(np.mean(per_batch_ms)) if per_batch_ms else None
    max_memory_mb = (observed_max_mem / (1024.0 ** 2)) if observed_max_mem > 0 else None

    return MeasureResult(
        batch_size=batch_size,
        amortized_latency_ms=amortized_latency_ms,
        max_memory_mb=max_memory_mb,
        status=status,
        note=note,
    )


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def make_plot(
    df: pd.DataFrame,
    *,
    out_png: Path,
    title: str,
    device_total_mem_mb: Optional[float],
) -> None:
    """
    Create a dual-axis plot:
      - Left Y: amortized latency (ms / image) [black solid line, no markers]
      - Right Y: memory usage (% of device) [dashed line, no markers]
      - Legends: "Latency" on left, "Memory Usage" on right
    """
    df_plot = df.copy()
    df_plot = df_plot[df_plot["status"] == "ok"].sort_values("batch_size")

    if df_plot.empty:
        print("No successful measurements to plot.")
        return

    x = df_plot["batch_size"].to_numpy()
    latency_ms = df_plot["amortized_latency_ms"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Latency line: black solid, no markers
    l1, = ax1.plot(x, latency_ms, color="black", linestyle="-", label="Latency")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Amortized Inference Latency (ms / image)")
    ax1.grid(visible=True, alpha=0.3)

    # Secondary axis: memory percentage
    ax2 = ax1.twinx()
    l2 = None
    if device_total_mem_mb is not None and "max_memory_mb" in df_plot:
        mem_mb = pd.to_numeric(df_plot["max_memory_mb"], errors="coerce").to_numpy()
        if np.isfinite(mem_mb).all():
            pct = (mem_mb / max(1e-9, device_total_mem_mb)) * 100.0
            # Memory usage line: dashed, no markers
            l2, = ax2.plot(x, pct, linestyle="--", color="black", label="Memory Usage")
            ax2.set_ylabel("GPU Memory Consumption (%)")

    # Place legends separately: left axis (latency), right axis (memory)
    if l1:
        ax1.legend(handles=[l1], loc="upper left")
    if l2:
        ax2.legend(handles=[l2], loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[plot] Saved: {out_png}")



# --------------------------------------------------------------------------------------
# CLI and main
# --------------------------------------------------------------------------------------

def parse_batch_sizes(arg: str) -> List[int]:
    """Parse comma/space-separated ints like '25,50,100' or '25 50 100'."""
    tokens = [t.strip() for t in arg.replace(",", " ").split()]
    vals: List[int] = []
    for t in tokens:
        if not t:
            continue
        v = int(t)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"Batch size must be positive: {v}")
        vals.append(v)
    if not vals:
        raise argparse.ArgumentTypeError("No batch sizes parsed.")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure ResNeXt101_64x4d latency & memory vs batch size and plot results."
    )
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing flat PNG files named '[class_idx]_[uid].png'.")
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=[25, 50, 100, 200, 400, 600, 800],
                        help="Comma/space-separated batch sizes, e.g. '25,50,100' (default: 25..800).")
    parser.add_argument("--max-batches", type=int, default=100,
                        help="Max batches to measure per batch size.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4).")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available.")
    parser.add_argument("--autocast", action="store_true",
                        help="Use torch.autocast(float16) on CUDA for faster inference (if supported).")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory to write results.csv and plot.png (default: current dir).")
    parser.add_argument("--plot-title", type=str, default="Latency & GPU Memory vs Batch Size (ResNeXt101_64x4d)",
                        help="Title for the plot.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results_latency_memory.csv"
    out_png = out_dir / "latency_memory_plot.pdf"  # extension drives format; name retained

    device = torch.device("cpu")
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")

    ds = build_dataset(args.image_dir)

    # Compute device memory for percentage plotting (if on CUDA)
    device_total_mem_mb: Optional[float] = None
    pin_memory = (device.type == "cuda")
    dtype: Optional[torch.dtype] = torch.float16 if (device.type == "cuda" and args.autocast) else None

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        device_total_mem_mb = float(props.total_memory) / (1024.0 ** 2)
        print(f"[env] CUDA device: {props.name} | total VRAM: {device_total_mem_mb:.1f} MB")
    else:
        print("[env] Using CPU")

    cfg = MeasureConfig(
        batch_sizes=args.batch_sizes,
        max_batches=max(1, args.max_batches),
        num_workers=max(0, args.num_workers),
        device=device,
        pin_memory=pin_memory,
        autocast=bool(args.autocast and device.type == "cuda"),
        dtype=dtype,
    )

    # Run measurements
    results: List[MeasureResult] = []
    print("batch_size, amortized_latency_ms, max_memory_mb, status, note")
    for bs in cfg.batch_sizes:
        r = measure_for_batch_size(ds=ds, batch_size=bs, cfg=cfg)
        results.append(r)
        print(f"{r.batch_size}, {r.amortized_latency_ms}, {r.max_memory_mb}, {r.status}, {r.note!r}")

    # Save table
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(out_csv, index=False)
    print(f"[csv] Saved: {out_csv}")

    # Make plot (only successful rows will be drawn)
    make_plot(df, out_png=out_png, title=args.plot_title, device_total_mem_mb=device_total_mem_mb)

    # Also dump JSON for programmatic consumption
    out_json = out_dir / "results_latency_memory.json"
    with open(out_json, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    print(f"[json] Saved: {out_json}")


if __name__ == "__main__":
    main()
