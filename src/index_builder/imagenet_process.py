#!/usr/bin/env python3
"""
Flatten ImageNet-1k train tarball into a single directory of PNGs named:
    [class_idx]_[UID].png

Now with exact-size random subsampling via reservoir sampling:
- First pass: count all eligible images (JPEG/JPG/PNG) across inner class tars.
- Compute k = int(total * subset_frac).
- Second pass: select exactly k images via reservoir sampling (reproducible with --seed) and write them.

ImageNet train tarball structure (canonical):
- ILSVRC2012_img_train.tar
    - n01440764.tar
        - n01440764_XXXX.JPEG
        - ...
    - n01443537.tar
    - ...
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import tarfile
from PIL import Image, ImageFile

# Allow loading slightly truncated files without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Flatten ImageNet-1k train tarball into a single PNG directory (with exact-size random subsampling)."
    )
    p.add_argument(
        "train_tar",
        type=Path,
        help="Path to the ImageNet-1k train tarball (e.g., ILSVRC2012_img_train.tar).",
    )
    p.add_argument(
        "out_dir",
        type=Path,
        help="Output directory; will be created if it does not exist.",
    )
    p.add_argument(
        "--synset-map",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON or TXT file defining synset-to-class-index mapping. "
            "JSON format: {\"n01440764\": 0, ...}. "
            "TXT format: one synset per line in class-index order (line 0 -> class 0, etc.)."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if a name collision occurs (default: skip existing).",
    )
    p.add_argument(
        "--subset-frac",
        type=float,
        default=0.25,
        help="Target fraction of images to include (exact size via reservoir). Must be in (0, 1]. Default: 0.25",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reservoir sampling. Default: 42",
    )
    return p.parse_args()


# ======================================================================================
# Mapping utilities
# ======================================================================================

def load_mapping(path: Path) -> Dict[str, int]:
    """
    Load synset->index mapping from JSON or TXT.

    Parameters
    ----------
    path : Path
        Path to mapping file.

    Returns
    -------
    Dict[str, int]
        Mapping from synset (e.g., 'n01440764') to class index (int).

    Raises
    ------
    ValueError
        If file extension unsupported or format invalid.
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("JSON mapping must be an object: {synset: index}.")
        # Normalize keys
        return {str(k).strip(): int(v) for k, v in data.items()}
    elif path.suffix.lower() in {".txt", ".lst"}:
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        return {syn: idx for idx, syn in enumerate(lines)}
    else:
        raise ValueError(f"Unsupported mapping file extension: {path.suffix}")


def derive_mapping_from_top_level(train_tar: Path) -> Dict[str, int]:
    """
    Derive synset->index mapping by scanning the top-level members of the train tarball.
    The train tarball contains ~1000 inner tar files named '<synset>.tar'.
    We sort synset names lexicographically and assign 0..N-1.

    Parameters
    ----------
    train_tar : Path
        Path to ImageNet train tarball.

    Returns
    -------
    Dict[str, int]
        Mapping synset->index.
    """
    synsets: List[str] = []
    with tarfile.open(train_tar, mode="r:*") as tf:
        for m in tf.getmembers():
            # Expect names like 'n01440764.tar'
            name = Path(m.name).name
            if not name.endswith(".tar"):
                continue
            syn = name[:-4]  # strip .tar
            if syn and syn not in synsets:
                synsets.append(syn)
    synsets.sort()
    return {syn: i for i, syn in enumerate(synsets)}


# ======================================================================================
# Iteration helpers
# ======================================================================================

def iter_inner_class_tars(tf: tarfile.TarFile) -> Iterable[Tuple[str, bytes]]:
    """
    Iterate over inner class tar files within the top-level tar.

    Parameters
    ----------
    tf : tarfile.TarFile
        Opened top-level tar file.

    Yields
    ------
    Tuple[str, bytes]
        (synset, raw bytes of the inner class tar)
    """
    for m in tf.getmembers():
        name = Path(m.name).name
        if not name.endswith(".tar"):
            continue
        synset = name[:-4]
        if not synset:
            continue
        # Extract the nested class tar as bytes (streamed)
        extracted = tf.extractfile(m)
        if extracted is None:
            continue
        yield synset, extracted.read()


def is_image_member(member: tarfile.TarInfo) -> bool:
    """
    Quick filter for likely image files.

    Parameters
    ----------
    member : tarfile.TarInfo
        Inner tar member.

    Returns
    -------
    bool
        True if member looks like an image (JPEG/JPG/PNG).
    """
    if not member.isfile():
        return False
    lower = member.name.lower()
    return lower.endswith(".jpeg") or lower.endswith(".jpg") or lower.endswith(".png")


def normalize_to_png(img: Image.Image) -> Image.Image:
    """
    Normalize a PIL Image to PNG-safe mode (convert to 'RGB').

    Parameters
    ----------
    img : Image.Image
        Input PIL Image.

    Returns
    -------
    Image.Image
        Converted image ready to save as PNG.
    """
    if img.mode not in ("RGB",):
        return img.convert("RGB")
    return img


def extract_uid(filename: str) -> str:
    """
    Extract UID from an ImageNet filename such as 'n01440764_10026.JPEG'.
    We define UID as the part after the first underscore, without the extension.

    Parameters
    ----------
    filename : str
        Base filename (no directory).

    Returns
    -------
    str
        UID (e.g., '10026'). If pattern not matched, returns the stem without synset.
    """
    base = Path(filename).name
    stem = base.rsplit(".", 1)[0]
    if "_" in stem:
        return stem.split("_", 1)[1]
    # Fallback: if no underscore, use whole stem
    return stem


# ======================================================================================
# Reservoir sampling core
# ======================================================================================

def count_total_images(train_tar: Path, mapping: Dict[str, int]) -> int:
    """
    First pass: count all eligible images across all mapped synsets.

    Parameters
    ----------
    train_tar : Path
        Path to outer train tar.
    mapping : Dict[str, int]
        Synset -> class_idx mapping; only mapped synsets are counted.

    Returns
    -------
    int
        Total number of eligible image members.
    """
    total = 0
    with tarfile.open(train_tar, mode="r:*") as tf:
        for synset, inner_bytes in iter_inner_class_tars(tf):
            if synset not in mapping:
                # keep logic aligned with extraction pass
                continue
            with tarfile.open(fileobj=io.BytesIO(inner_bytes), mode="r:*") as inner_tf:
                for m in inner_tf.getmembers():
                    if is_image_member(m):
                        total += 1
    return total


def reservoir_sample_indices(n: int, k: int, seed: int) -> List[int]:
    """
    Standard single-pass reservoir sampling to select k indices from range(n).

    Parameters
    ----------
    n : int
        Stream length.
    k : int
        Number of samples to draw (0 <= k <= n).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    List[int]
        Sorted list of chosen indices in [0, n).
    """
    rng = random.Random(seed)
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))  # degenerate case

    reservoir = list(range(k))
    for i in range(k, n):
        j = rng.randint(0, i)  # inclusive ends
        if j < k:
            reservoir[j] = i
    reservoir.sort()
    return reservoir


# ======================================================================================
# Extraction pass (exact subset via reservoir)
# ======================================================================================

def process_exact_subset(
    *,
    train_tar: Path,
    out_dir: Path,
    mapping: Dict[str, int],
    chosen_indices: List[int],
    force: bool,
) -> Dict[str, int]:
    """
    Second pass: iterate images in the same deterministic order as counting and
    write only those whose linear index is in chosen_indices.

    Parameters
    ----------
    train_tar : Path
        Path to outer train tar.
    out_dir : Path
        Output directory root.
    mapping : Dict[str, int]
        Synset -> class_idx.
    chosen_indices : List[int]
        Sorted list of selected linear indices (0..N-1) across all images.
    force : bool
        Overwrite existing files if True.

    Returns
    -------
    Dict[str, int]
        Counters: {'written', 'skipped', 'errors'}
    """
    counters = {"written": 0, "skipped": 0, "errors": 0}

    # Two-pointer sweep over the linear stream index and chosen indices
    target_set = set(chosen_indices)  # membership O(1)
    cur_idx = -1

    with tarfile.open(train_tar, mode="r:*") as tf:
        for synset, inner_bytes in iter_inner_class_tars(tf):
            if synset not in mapping:
                continue
            class_idx = mapping[synset]
            try:
                with tarfile.open(fileobj=io.BytesIO(inner_bytes), mode="r:*") as inner_tf:
                    for m in inner_tf.getmembers():
                        if not is_image_member(m):
                            continue

                        cur_idx += 1
                        if cur_idx not in target_set:
                            continue  # not selected by reservoir

                        f = inner_tf.extractfile(m)
                        if f is None:
                            counters["errors"] += 1
                            continue
                        try:
                            with Image.open(f) as im:
                                im = normalize_to_png(im)
                                uid = extract_uid(Path(m.name).name)
                                out_path = out_dir / f"{class_idx}_{uid}.png"
                                if out_path.exists() and not force:
                                    counters["skipped"] += 1
                                else:
                                    out_path.parent.mkdir(parents=True, exist_ok=True)
                                    im.save(out_path, format="PNG", optimize=True)
                                    counters["written"] += 1
                        except Exception as e:
                            counters["errors"] += 1
                            print(f"[WARN] Failed {synset}:{m.name}: {e}", file=sys.stderr)
            except Exception as e:
                counters["errors"] += 1
                print(f"[WARN] Failed to open inner tar for {synset}: {e}", file=sys.stderr)

    return counters


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    """
    CLI entrypoint: orchestrates mapping, two-pass reservoir sampling, and conversion.
    """
    args = parse_args()

    # Basic checks
    if not args.train_tar.is_file():
        print(f"[ERR] Train tar not found: {args.train_tar}", file=sys.stderr)
        sys.exit(2)

    if not (0.0 < args.subset_frac <= 1.0):
        print(f"[ERR] --subset-frac must be in (0, 1]; got {args.subset_frac}", file=sys.stderr)
        sys.exit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build synset->index mapping
    if args.synset_map:
        mapping = load_mapping(args.synset_map)
        print(f"[INFO] Loaded mapping for {len(mapping)} synsets from {args.synset_map}")
    else:
        print("[INFO] No mapping provided; deriving from tarball (sorted synsets → indices).")
        mapping = derive_mapping_from_top_level(args.train_tar)
        print(f"[INFO] Derived mapping for {len(mapping)} synsets.")

    if len(mapping) == 0:
        print("[ERR] No synsets found/mapped. Aborting.", file=sys.stderr)
        sys.exit(3)

    # Pass 1: Count total eligible images
    print("[INFO] Counting total eligible images (pass 1/2)...")
    total_images = count_total_images(args.train_tar, mapping)
    print(f"[INFO] Found {total_images} eligible images in mapped synsets.")

    if total_images == 0:
        print("[ERR] No eligible images found. Aborting.", file=sys.stderr)
        sys.exit(3)

    # Compute exact target sample size k
    k = int(total_images * args.subset_frac)
    if k <= 0 and args.subset_frac > 0:
        # Ensure at least 1 image if user asked for a positive fraction
        k = 1
    if k > total_images:
        k = total_images

    print(f"[INFO] Subset fraction: {args.subset_frac:.6f} → target k = {k} images "
          f"(of total N = {total_images}). Seed = {args.seed}")

    # Draw reservoir indices deterministically
    chosen_indices = reservoir_sample_indices(total_images, k, args.seed)
    assert len(chosen_indices) == k, "Internal error: reservoir size mismatch."

    # Pass 2: Extract/write exactly the chosen indices
    print("[INFO] Extracting sampled images (pass 2/2)...")
    counters = process_exact_subset(
        train_tar=args.train_tar,
        out_dir=args.out_dir,
        mapping=mapping,
        chosen_indices=chosen_indices,
        force=bool(args.force),
    )

    print(
        f"[DONE] Wrote: {counters['written']} PNGs; "
        f"skipped existing: {counters['skipped']}; "
        f"errors: {counters['errors']}."
    )
    print(f"[OUT] Directory: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
