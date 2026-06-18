"""
core.py - Remote sensing image tiling core logic

Responsibility: pure computation, no UI framework dependency
Callers: app.py (Streamlit) / cli.py (command line) / any Python script
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import os
import csv
import json
import hashlib
import multiprocessing as mp
from functools import partial
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict
import gc
import time
import logging
import shutil

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class CropConfig:
    """All parameters for a single tiling task."""
    src_path: str
    output_dir: str
    tile_size: int = 2000
    overlap: float = 0.1
    edge_threshold: float = 0.9
    num_workers: Optional[int] = None
    suffix: Optional[str] = None
    single_mode: bool = False
    label_path: Optional[str] = None
    label_nodata: Optional[str] = None
    img_nodata: Optional[str] = None
    resume: bool = False

    def params_hash(self) -> str:
        """Generate a parameter fingerprint for resume validation."""
        h = hashlib.md5()
        h.update(str(self.src_path).encode())
        h.update(str(self.label_path or "").encode())
        h.update(str(self.tile_size).encode())
        h.update(str(self.overlap).encode())
        h.update(str(self.edge_threshold).encode())
        h.update(str(self.single_mode).encode())
        h.update(str(self.label_nodata).encode())
        h.update(str(self.img_nodata).encode())
        h.update(str(self.suffix).encode())
        return h.hexdigest()[:12]


@dataclass
class BatchItem:
    """One image-pair entry in a batch tiling job."""
    src_path: str
    label_path: Optional[str] = None
    suffix: Optional[str] = None


@dataclass
class TileRecord:
    """Output record for a single tile."""
    tile_id: str
    image_fn: str
    label_fn: Optional[str] = None


@dataclass
class CropResult:
    """Complete result of a tiling task."""
    total_tasks: int = 0
    valid_tiles: int = 0
    skipped_nodata: int = 0
    skipped_small: int = 0
    skipped_existing: int = 0
    errors: int = 0
    elapsed_time: float = 0.0
    csv_path: str = ""
    tile_records: List[Dict[str, str]] = field(default_factory=list)

    @property
    def skip_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.skipped_nodata + self.skipped_small + self.errors) / self.total_tasks


@dataclass
class RasterInfo:
    """Raster metadata (for preview / display)."""
    path: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs: Optional[str]
    nodata: Optional[float]
    nodatavals: Optional[Tuple]
    resolution: Optional[Tuple[float, float]] = None
    has_overviews: bool = False


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def parse_nodata(nodata_str: Optional[str], band_count: int) -> Optional[Tuple]:
    """
    Parse a NoData string into a tuple matching the raster's band count.

    Examples:
        parse_nodata("256", 4)             -> (256, 256, 256, 256)
        parse_nodata("256,256,256,256", 4) -> (256, 256, 256, 256)
    """
    if nodata_str is None:
        return None
    try:
        vals = [float(v) if '.' in v else int(v) for v in nodata_str.split(',')]
        if len(vals) == 1:
            return tuple(vals * band_count)
        elif len(vals) == band_count:
            return tuple(vals)
        else:
            raise ValueError(
                f"NoData value count ({len(vals)}) does not match band count ({band_count})"
            )
    except ValueError as e:
        raise ValueError(f"Invalid NoData value: {e}")


def pad_to_size(data: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Pad array to target size using edge values (replaces original for-loop).

    Args:
        data: Array with shape (bands, height, width)
        target_height: Target height
        target_width: Target width

    Returns:
        Padded array with shape (bands, target_height, target_width)
    """
    bands, h, w = data.shape
    if h == target_height and w == target_width:
        return data

    pad_h = target_height - h
    pad_w = target_width - w

    if pad_h < 0 or pad_w < 0:
        return data[:, :target_height, :target_width]

    pad_width = ((0, 0), (0, pad_h), (0, pad_w))
    return np.pad(data, pad_width, mode='edge')


def get_raster_info(path: str) -> RasterInfo:
    """Read raster metadata without loading full data."""
    with rasterio.open(path) as src:
        res = src.res if src.res else None
        has_ov = len(src.overviews(1)) > 0 if src.count > 0 else False
        return RasterInfo(
            path=path,
            width=src.width,
            height=src.height,
            band_count=src.count,
            dtype=str(src.dtypes[0]),
            crs=str(src.crs) if src.crs else None,
            nodata=src.nodata,
            nodatavals=src.nodatavals,
            resolution=res,
            has_overviews=has_ov,
        )


def scan_tiles(config: CropConfig) -> List[Tuple]:
    """
    Scan the raster and generate all tile positions (no actual cropping).
    Returns a task list for execute_crop to consume directly.
    """
    step = int(config.tile_size * (1 - config.overlap))

    if config.single_mode:
        with rasterio.open(config.src_path) as src:
            h, w = src.height, src.width
        return [(x, y, config.tile_size, step, config.src_path, str(config.output_dir))
                for y in range(0, h, step) for x in range(0, w, step)]
    else:
        if not config.label_path:
            raise ValueError("Label path is required in dual mode")
        with rasterio.open(config.label_path) as src:
            h, w = src.height, src.width
        return [(x, y, config.tile_size, step, config.label_path, config.src_path, str(config.output_dir))
                for y in range(0, h, step) for x in range(0, w, step)]


# ──────────────────────────────────────────────
# Checkpoint (resume support)
# ──────────────────────────────────────────────

CHECKPOINT_FILE = "_clips_checkpoint.json"


def _checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, CHECKPOINT_FILE)


def save_checkpoint(output_dir: str, params_hash: str, completed_ids: List[str]):
    """Save a checkpoint for resume."""
    cp = {
        "params_hash": params_hash,
        "completed_ids": sorted(completed_ids),
        "timestamp": time.time(),
    }
    with open(_checkpoint_path(output_dir), 'w') as f:
        json.dump(cp, f)


def load_checkpoint(output_dir: str, params_hash: str) -> Optional[set]:
    """
    Load a checkpoint, returning the set of completed tile IDs.
    Returns None if the parameter fingerprint does not match.
    """
    cp_path = _checkpoint_path(output_dir)
    if not os.path.exists(cp_path):
        return None
    try:
        with open(cp_path, 'r') as f:
            cp = json.load(f)
        if cp.get("params_hash") != params_hash:
            logger.warning("Checkpoint parameter mismatch; ignoring stale checkpoint")
            return None
        return set(cp.get("completed_ids", []))
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Corrupted checkpoint file: {e}; ignoring")
        return None


def remove_checkpoint(output_dir: str):
    """Remove checkpoint file after task completion."""
    cp_path = _checkpoint_path(output_dir)
    if os.path.exists(cp_path):
        os.remove(cp_path)


# ──────────────────────────────────────────────
# Tile processing functions
# ──────────────────────────────────────────────

def process_single_tile(task, user_nodata=None, suffix=None, edge_threshold=0.9,
                        skip_ids: Optional[set] = None):
    """
    Process a single tile (single-image mode).

    Returns:
        dict: {"tile_id", "image_fn", "label_fn", "skip_reason"}
              skip_reason is None on success, otherwise describes why it was skipped.
    """
    x, y, tile_size, step, src_path, output_dir = task

    tile_id = f"tile_{y // step}_{x // step}"

    if skip_ids and tile_id in skip_ids:
        return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                "skip_reason": "existing"}

    try:
        output_subdir = os.path.join(output_dir, "images")
        Path(output_subdir).mkdir(exist_ok=True, parents=True)

        with rasterio.open(src_path) as src:
            height, width = src.height, src.width

            actual_width = min(tile_size, width - x)
            actual_height = min(tile_size, height - y)
            window = Window(x, y, actual_width, actual_height)

            if window.width < tile_size * edge_threshold or \
               window.height < tile_size * edge_threshold:
                return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                        "skip_reason": "small"}

            # Read data + profile in a single file open (fixes original double-open issue)
            tile_data = src.read(window=window)
            profile = src.profile.copy()
            original_transform = src.window_transform(window)

            if user_nodata is not None:
                nodatavals = user_nodata
            else:
                nodatavals = (src.nodatavals
                              if src.nodatavals and all(v is not None for v in src.nodatavals)
                              else None)

            if nodatavals:
                nodata_array = np.array(nodatavals)[:, None, None]
                if np.any(np.all(tile_data == nodata_array, axis=0)):
                    return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                            "skip_reason": "nodata"}

            if tile_data.shape[1] < tile_size or tile_data.shape[2] < tile_size:
                tile_data = pad_to_size(tile_data, tile_size, tile_size)

        output_path = os.path.join(output_subdir, f"{suffix or ''}{tile_id}_label.tif")
        profile.update({
            "height": tile_size,
            "width": tile_size,
            "transform": original_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        })
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(tile_data)

        gc.collect()
        return {"tile_id": tile_id,
                "image_fn": os.path.join("./images", f"{suffix or ''}{tile_id}_label.tif"),
                "label_fn": None,
                "skip_reason": None}

    except Exception as e:
        logger.error(f"Tile {tile_id} ({x},{y}) failed: {e}")
        return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                "skip_reason": "error"}


def process_dual_tile(task, label_nodata=None, img_nodata=None, suffix=None,
                      edge_threshold=0.9, skip_ids: Optional[set] = None):
    """
    Process a single tile (dual mode) - synchronized label and image cropping.

    Returns:
        dict: {"tile_id", "image_fn", "label_fn", "skip_reason"}
    """
    x, y, tile_size, step, label_src_path, img_src_path, output_dir = task
    tile_id = f"tile_{y // step}_{x // step}"

    if skip_ids and tile_id in skip_ids:
        return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                "skip_reason": "existing"}

    try:
        hr_img_dir = os.path.join(output_dir, "HR_img")
        lr_label_dir = os.path.join(output_dir, "LR_label")
        Path(hr_img_dir).mkdir(exist_ok=True, parents=True)
        Path(lr_label_dir).mkdir(exist_ok=True, parents=True)

        # -- Label tile --
        with rasterio.open(label_src_path) as label_src:
            height, width = label_src.height, label_src.width
            actual_width = min(tile_size, width - x)
            actual_height = min(tile_size, height - y)
            label_window = Window(x, y, actual_width, actual_height)

            if label_window.width < tile_size * edge_threshold or \
               label_window.height < tile_size * edge_threshold:
                return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                        "skip_reason": "small"}

            label_tile = label_src.read(window=label_window)
            label_profile = label_src.profile.copy()
            label_transform = label_src.window_transform(label_window)
            label_bounds = label_src.window_bounds(label_window)

            if label_nodata is not None:
                nodata_val = label_nodata[0]
            else:
                nodata_val = label_src.nodata

            if nodata_val is not None and np.any(label_tile == nodata_val):
                return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                        "skip_reason": "nodata"}

            label_tile = pad_to_size(label_tile, tile_size, tile_size)

        # -- Image tile --
        with rasterio.open(img_src_path) as img_src:
            img_window = rasterio.windows.from_bounds(
                *label_bounds, transform=img_src.transform
            )
            img_window = rasterio.windows.intersection(
                img_window, Window(0, 0, img_src.width, img_src.height)
            )

            if img_window.width <= 0 or img_window.height <= 0:
                return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                        "skip_reason": "small"}

            img_tile = img_src.read(window=img_window)
            img_profile = img_src.profile.copy()
            img_transform = img_src.window_transform(img_window)

            if img_nodata is not None:
                img_nodata_mask = np.all(
                    img_tile == np.array(img_nodata)[:, None, None], axis=0
                )
                if np.any(img_nodata_mask):
                    return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                            "skip_reason": "nodata"}

            target_img_height = int(tile_size * (img_window.height / label_window.height))
            target_img_width = int(tile_size * (img_window.width / label_window.width))
            if img_tile.shape[1] < target_img_height or img_tile.shape[2] < target_img_width:
                img_tile = pad_to_size(img_tile, target_img_height, target_img_width)

        # -- Write --
        label_out = os.path.join(lr_label_dir, f"{suffix or ''}{tile_id}_label.tif")
        img_out = os.path.join(hr_img_dir, f"{suffix or ''}{tile_id}_image.tif")

        label_profile.update({
            "height": tile_size, "width": tile_size,
            "transform": label_transform,
            "compress": "lzw", "tiled": True,
            "blockxsize": 256, "blockysize": 256,
        })
        with rasterio.open(label_out, 'w', **label_profile) as dst:
            dst.write(label_tile)

        img_profile.update({
            "height": img_tile.shape[1], "width": img_tile.shape[2],
            "transform": img_transform,
            "compress": "lzw", "tiled": True,
            "blockxsize": 256, "blockysize": 256,
        })
        with rasterio.open(img_out, 'w', **img_profile) as dst:
            dst.write(img_tile)

        gc.collect()
        return {"tile_id": tile_id,
                "image_fn": os.path.join("./HR_img", f"{suffix or ''}{tile_id}_image.tif"),
                "label_fn": os.path.join("./LR_label", f"{suffix or ''}{tile_id}_label.tif"),
                "skip_reason": None}

    except Exception as e:
        logger.error(f"Tile {tile_id} ({x},{y}) failed: {e}")
        return {"tile_id": tile_id, "image_fn": None, "label_fn": None,
                "skip_reason": "error"}


# ──────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────

def execute_crop(
    config: CropConfig,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> CropResult:
    """
    Execute a tiling task.

    Args:
        config: Tiling configuration
        progress_callback: Progress callback (completed, total, tile_id)

    Returns:
        CropResult: Tiling result statistics
    """
    start_time = time.time()
    result = CropResult()
    output_dir = str(config.output_dir)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    num_workers = config.num_workers or max(1, mp.cpu_count() - 1)

    logger.info(f"CPUs available: {mp.cpu_count()}, using {num_workers} workers")
    logger.info(f"Mode: {'Single' if config.single_mode else 'Dual'} | "
                f"Tile: {config.tile_size}x{config.tile_size} | "
                f"Overlap: {config.overlap:.0%}")

    # -- Resume --
    skip_ids = None
    if config.resume:
        skip_ids = load_checkpoint(output_dir, config.params_hash())
        if skip_ids:
            logger.info(f"Resume: skipping {len(skip_ids)} completed tiles")
        else:
            logger.info("No valid checkpoint found; starting from scratch")

    # -- Parse NoData --
    img_nodata_parsed = None
    label_nodata_parsed = None

    if config.single_mode:
        with rasterio.open(config.src_path) as src:
            img_nodata_parsed = parse_nodata(config.img_nodata, src.count) \
                                if config.img_nodata else None
    else:
        with rasterio.open(config.label_path) as label_src:
            label_nodata_parsed = parse_nodata(config.label_nodata, label_src.count) \
                                  if config.label_nodata else None
            with rasterio.open(config.src_path) as img_src:
                if label_src.crs != img_src.crs:
                    raise ValueError("Label and image CRS do not match")
                if not np.allclose(
                    [label_src.transform.a, label_src.transform.e],
                    [img_src.transform.a, img_src.transform.e],
                    rtol=0.001
                ):
                    raise ValueError("Label and image spatial transforms are inconsistent")
                img_nodata_parsed = parse_nodata(config.img_nodata, img_src.count) \
                                    if config.img_nodata else None

    # -- Generate task list --
    tasks = scan_tiles(config)
    result.total_tasks = len(tasks)
    logger.info(f"Generated {len(tasks)} tile positions")

    # -- Periodic checkpoint saving --
    completed_ids = list(skip_ids) if skip_ids else []
    checkpoint_counter = 0

    def _on_result(r):
        nonlocal checkpoint_counter
        if r["skip_reason"] is None:
            completed_ids.append(r["tile_id"])
            checkpoint_counter += 1
            if checkpoint_counter % 50 == 0:
                save_checkpoint(output_dir, config.params_hash(), completed_ids)

    # -- Execute --
    with rasterio.Env(GDAL_CACHEMAX=512):
        if config.single_mode:
            process_func = partial(
                process_single_tile,
                user_nodata=img_nodata_parsed,
                suffix=config.suffix,
                edge_threshold=config.edge_threshold,
                skip_ids=skip_ids,
            )
        else:
            process_func = partial(
                process_dual_tile,
                label_nodata=label_nodata_parsed,
                img_nodata=img_nodata_parsed,
                suffix=config.suffix,
                edge_threshold=config.edge_threshold,
                skip_ids=skip_ids,
            )

        with mp.Pool(processes=num_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_func, tasks)):
                _on_result(r)

                if r["skip_reason"] is None:
                    result.valid_tiles += 1
                elif r["skip_reason"] == "nodata":
                    result.skipped_nodata += 1
                elif r["skip_reason"] == "small":
                    result.skipped_small += 1
                elif r["skip_reason"] == "existing":
                    result.skipped_existing += 1
                elif r["skip_reason"] == "error":
                    result.errors += 1

                if progress_callback:
                    progress_callback(i + 1, result.total_tasks, r["tile_id"])

    # -- Generate CSV --
    valid_records = _scan_output_dir(output_dir, config.single_mode)
    result.valid_tiles = len(valid_records)

    csv_path = os.path.join(output_dir, "tile_records.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_fn'] if config.single_mode else ['image_fn', 'label_fn']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in valid_records:
            writer.writerow(rec)
    result.csv_path = csv_path

    # -- Clean up checkpoint --
    remove_checkpoint(output_dir)

    result.elapsed_time = time.time() - start_time
    logger.info(
        f"Done: {result.valid_tiles} valid tiles / {result.total_tasks} total | "
        f"NoData skipped: {result.skipped_nodata} | Too small: {result.skipped_small} | "
        f"Errors: {result.errors} | Time: {result.elapsed_time:.1f}s"
    )

    return result


def _scan_output_dir(output_dir: str, single_mode: bool) -> List[Dict[str, str]]:
    """Scan output directory and build valid tile records."""
    valid_records = []
    if single_mode:
        images_dir = os.path.join(output_dir, "images")
        if os.path.isdir(images_dir):
            for fn in sorted(os.listdir(images_dir)):
                if fn.endswith(".tif"):
                    valid_records.append({"image_fn": os.path.join("./images", fn)})
    else:
        hr_dir = os.path.join(output_dir, "HR_img")
        lr_dir = os.path.join(output_dir, "LR_label")
        if os.path.isdir(hr_dir) and os.path.isdir(lr_dir):
            hr_files = {os.path.splitext(f)[0]: f for f in os.listdir(hr_dir) if f.endswith(".tif")}
            lr_files = {os.path.splitext(f)[0]: f for f in os.listdir(lr_dir) if f.endswith(".tif")}
            for key in sorted(hr_files.keys()):
                if key in lr_files:
                    valid_records.append({
                        "image_fn": os.path.join("./HR_img", hr_files[key]),
                        "label_fn": os.path.join("./LR_label", lr_files[key]),
                    })
    return valid_records


# ──────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────

def execute_batch(
    items: List[BatchItem],
    output_dir: str,
    tile_size: int = 2000,
    overlap: float = 0.1,
    edge_threshold: float = 0.9,
    num_workers: Optional[int] = None,
    single_mode: bool = False,
    label_nodata: Optional[str] = None,
    img_nodata: Optional[str] = None,
    resume: bool = False,
    progress_callback: Optional[Callable[[int, int, str, int, int], None]] = None,
) -> List[Tuple[BatchItem, CropResult]]:
    """
    Execute a batch of tiling tasks. Each item gets its own subdirectory
    under output_dir named after the source file stem.

    Args:
        items: List of BatchItem (src_path, label_path, suffix)
        output_dir: Root output directory
        tile_size, overlap, edge_threshold, num_workers: Shared params
        single_mode: Whether all items are single-mode
        label_nodata, img_nodata: Shared NoData strings
        resume: Whether to resume from checkpoints
        progress_callback: (item_idx, total_items, tile_id, tiles_done, tiles_total)

    Returns:
        List of (BatchItem, CropResult) tuples
    """
    results = []
    for idx, item in enumerate(items):
        # Derive subdirectory name from source file
        stem = Path(item.src_path).stem
        item_output_dir = os.path.join(output_dir, stem)

        config = CropConfig(
            src_path=item.src_path,
            output_dir=item_output_dir,
            tile_size=tile_size,
            overlap=overlap,
            edge_threshold=edge_threshold,
            num_workers=num_workers,
            suffix=item.suffix,
            single_mode=single_mode,
            label_path=item.label_path,
            label_nodata=label_nodata,
            img_nodata=img_nodata,
            resume=resume,
        )

        # Wrap progress callback to include batch context
        def _batch_progress(done, total, tile_id, _idx=idx, _n=len(items)):
            if progress_callback:
                progress_callback(_idx, _n, tile_id, done, total)

        logger.info(f"Batch [{idx+1}/{len(items)}] Processing: {item.src_path}")
        result = execute_crop(config, progress_callback=_batch_progress)
        results.append((item, result))

    return results


def merge_batch_csvs(output_dir: str, single_mode: bool) -> str:
    """
    Merge all per-item tile_records.csv into a single root-level CSV.

    Returns:
        Path to the merged CSV file.
    """
    merged_path = os.path.join(output_dir, "tile_records_all.csv")
    fieldnames = ['image_fn'] if single_mode else ['image_fn', 'label_fn']

    with open(merged_path, 'w', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in sorted(os.listdir(output_dir)):
            sub = os.path.join(output_dir, entry)
            csv_file = os.path.join(sub, "tile_records.csv")
            if os.path.isdir(sub) and os.path.isfile(csv_file):
                with open(csv_file, 'r') as in_f:
                    reader = csv.DictReader(in_f)
                    for row in reader:
                        # Prepend subdirectory to paths
                        prefixed = {}
                        for k, v in row.items():
                            parts = v.split("/", 1)
                            if len(parts) == 2:
                                prefixed[k] = f"{entry}/{parts[0]}/{parts[1]}"
                            else:
                                prefixed[k] = f"{entry}/{v}"
                        writer.writerow(prefixed)

    return merged_path


# ──────────────────────────────────────────────
# Training format packaging
# ──────────────────────────────────────────────

def package_as_dataset(
    output_dir: str,
    single_mode: bool,
    format_type: str = "torch",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> str:
    """
    Package tiling output into a training-ready format.

    Supported formats:
        - "torch":  PyTorch Dataset index (train.txt / val.txt / test.txt with file lists)
        - "hf":     HuggingFace Dataset structure (metadata.jsonl + README.md)

    Args:
        output_dir: Root output directory (batch or single)
        single_mode: Whether single-image mode
        format_type: "torch" or "hf"
        train_ratio: Train split ratio
        val_ratio: Validation split ratio (test = 1 - train - val)
        seed: Random seed for reproducible splits

    Returns:
        Path to the generated index/metadata file.
    """
    rng = np.random.RandomState(seed)

    # Collect all valid tile records
    all_records = _scan_output_dir(output_dir, single_mode)

    # Check for batch subdirectories
    for entry in sorted(os.listdir(output_dir)):
        sub = os.path.join(output_dir, entry)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "tile_records.csv")):
            for rec in _scan_output_dir(sub, single_mode):
                prefixed = {}
                for k, v in rec.items():
                    prefixed[k] = f"{entry}/{v}"
                all_records.append(prefixed)

    if not all_records:
        logger.warning("No valid tile records found for packaging")
        return ""

    # Shuffle and split
    indices = rng.permutation(len(all_records))
    n_train = int(len(all_records) * train_ratio)
    n_val = int(len(all_records) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        "train": [all_records[i] for i in train_idx],
        "val": [all_records[i] for i in val_idx],
        "test": [all_records[i] for i in test_idx],
    }

    if format_type == "torch":
        return _package_torch(output_dir, splits, single_mode)
    elif format_type == "hf":
        return _package_hf(output_dir, splits, single_mode)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def _package_torch(output_dir: str, splits: Dict[str, List[Dict]], single_mode: bool) -> str:
    """Generate PyTorch-style file lists: train.txt / val.txt / test.txt"""
    index_dir = os.path.join(output_dir, "dataset_index")
    os.makedirs(index_dir, exist_ok=True)

    for split_name, records in splits.items():
        split_path = os.path.join(index_dir, f"{split_name}.txt")
        with open(split_path, 'w') as f:
            for rec in records:
                if single_mode:
                    f.write(f"{rec['image_fn']}\n")
                else:
                    f.write(f"{rec['image_fn']} {rec['label_fn']}\n")

    # Write a simple PyTorch Dataset loader
    loader_path = os.path.join(index_dir, "tile_dataset.py")
    loader_code = '''"""
tile_dataset.py - PyTorch Dataset for tiled remote sensing data
Auto-generated by rs-clips
"""
import os
import numpy as np
import rasterio
from torch.utils.data import Dataset


class TileDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir: Root output directory from rs-clips
            split: One of "train", "val", "test"
            transform: Optional transform to apply to image tiles
        """
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []

        index_file = os.path.join(root_dir, "dataset_index", f"{split}.txt")
        with open(index_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.pairs.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    self.pairs.append((parts[0], None))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.pairs[idx][0])
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)

        # Normalize to [0, 1]
        image = image / image.max() if image.max() > 0 else image

        if self.pairs[idx][1] is not None:
            label_path = os.path.join(self.root_dir, self.pairs[idx][1])
            with rasterio.open(label_path) as src:
                label = src.read().astype(np.int64)
            if self.transform:
                image = self.transform(image)
            return image, label
        else:
            if self.transform:
                image = self.transform(image)
            return image
'''
    with open(loader_path, 'w') as f:
        f.write(loader_code)

    logger.info(f"PyTorch dataset index written to {index_dir}")
    return index_dir


def _package_hf(output_dir: str, splits: Dict[str, List[Dict]], single_mode: bool) -> str:
    """Generate HuggingFace Dataset structure: metadata.jsonl per split + README.md"""
    import json as _json

    hf_dir = os.path.join(output_dir, "hf_dataset")
    os.makedirs(hf_dir, exist_ok=True)

    for split_name, records in splits.items():
        jsonl_path = os.path.join(hf_dir, f"{split_name}.jsonl")
        with open(jsonl_path, 'w') as f:
            for rec in records:
                entry = {"image_fn": rec["image_fn"]}
                if not single_mode and "label_fn" in rec:
                    entry["label_fn"] = rec["label_fn"]
                f.write(_json.dumps(entry) + "\n")

    # README with dataset card
    readme = f"""---
task_categories:
  - image-classification
  - semantic-segmentation
tags:
  - remote-sensing
  - geospatial
size_categories:
  - {'1K<n<10K' if sum(len(v) for v in splits.values()) < 10000 else '10K<n<100K'}
---

# Remote Sensing Tile Dataset

Auto-generated by [rs-clips](https://github.com/zplzmzmpl/rs-clips).

## Splits

| Split   | Count |
|---------|-------|
| train   | {len(splits['train'])} |
| val     | {len(splits['val'])} |
| test    | {len(splits['test'])} |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{os.path.basename(output_dir)}")
```

## Structure

Each record contains:
- `image_fn`: Relative path to the image tile
{'- `label_fn`: Relative path to the label tile' if not single_mode else ''}
"""
    with open(os.path.join(hf_dir, "README.md"), 'w') as f:
        f.write(readme)

    logger.info(f"HuggingFace dataset written to {hf_dir}")
    return hf_dir
