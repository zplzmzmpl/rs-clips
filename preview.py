"""
preview.py - Tile preview generation

Features:
1. Thumbnail generation (using GDAL overview or downsampled read)
2. Tile grid overlay
3. NoData area highlighting
4. Statistics estimation (valid tile count, disk usage)
"""

import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

from core import CropConfig, parse_nodata, scan_tiles, get_raster_info

logger = logging.getLogger(__name__)

THUMBNAIL_MAX = 800


def generate_thumbnail(
    path: str,
    max_size: int = THUMBNAIL_MAX,
    nodata: Optional[Tuple] = None,
) -> Tuple[np.ndarray, float]:
    """
    Generate a raster thumbnail.

    Uses GDAL overview when available (near-zero I/O), otherwise downsampled read.
    For multi-band rasters, the first 3 bands are used as RGB; single-band is auto-stretched.

    Args:
        path: Raster file path
        max_size: Maximum thumbnail edge length (pixels)
        nodata: User-specified NoData value tuple

    Returns:
        (thumbnail, scale): thumbnail array (H, W, 3) uint8, scale factor
    """
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = max_size / max(h, w)
        if scale >= 1.0:
            scale = 1.0

        out_h = max(1, int(h * scale))
        out_w = max(1, int(w * scale))

        bands_to_read = list(range(1, min(src.count + 1, 4)))
        if len(bands_to_read) == 1:
            bands_to_read = [1]

        try:
            data = src.read(
                bands_to_read,
                out_shape=(len(bands_to_read), out_h, out_w),
                resampling=rasterio.enums.Resampling.average,
            )
        except Exception:
            data = src.read(bands_to_read)

        nodata_val = None
        if nodata is not None:
            nodata_val = nodata[0]
        elif src.nodata is not None:
            nodata_val = src.nodata

    thumbnail = _normalize_to_uint8(data, nodata_val)

    if thumbnail.shape[0] == 1:
        thumbnail = np.repeat(thumbnail, 3, axis=0)

    if thumbnail.shape[0] < 3:
        pad_ch = 3 - thumbnail.shape[0]
        thumbnail = np.concatenate(
            [thumbnail, np.zeros((pad_ch, thumbnail.shape[1], thumbnail.shape[2]), dtype=np.uint8)],
            axis=0,
        )
    elif thumbnail.shape[0] > 3:
        thumbnail = thumbnail[:3]

    thumbnail = np.transpose(thumbnail, (1, 2, 0))

    return thumbnail, scale


def _normalize_to_uint8(data: np.ndarray, nodata_val=None) -> np.ndarray:
    """Normalize arbitrary-dtype band data to 0-255 uint8."""
    result = np.zeros_like(data, dtype=np.uint8)

    for i in range(data.shape[0]):
        band = data[i].astype(np.float64)

        if nodata_val is not None:
            valid = band != nodata_val
        else:
            valid = np.ones_like(band, dtype=bool)

        if not np.any(valid):
            continue

        vmin = np.min(band[valid])
        vmax = np.max(band[valid])

        if vmax == vmin:
            result[i] = np.where(valid, 128, 0).astype(np.uint8)
        else:
            stretched = (band - vmin) / (vmax - vmin) * 255.0
            result[i] = np.where(valid, stretched, 0).astype(np.uint8)

    return result


def generate_nodata_heatmap(
    path: str,
    nodata: Optional[Tuple] = None,
    grid_size: int = 100,
    max_size: int = THUMBNAIL_MAX,
) -> Tuple[np.ndarray, float]:
    """
    Generate a NoData area heatmap.

    Args:
        path: Raster file path
        nodata: NoData value tuple
        grid_size: Grid granularity
        max_size: Maximum edge length

    Returns:
        (heatmap, scale): RGBA overlay (H, W, 4) float [0,1], scale factor
    """
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = max_size / max(h, w)
        if scale >= 1.0:
            scale = 1.0
        out_h = max(1, int(h * scale))
        out_w = max(1, int(w * scale))

        data = src.read(
            out_shape=(src.count, out_h, out_w),
            resampling=rasterio.enums.Resampling.nearest,
        )

        if nodata is not None:
            nodata_array = np.array(nodata)[:, None, None]
        elif src.nodata is not None:
            nodata_array = np.array([src.nodata] * src.count)[:, None, None]
        else:
            return np.zeros((out_h, out_w, 4), dtype=np.float32), scale

    nodata_mask = np.all(data == nodata_array, axis=0)

    heatmap = np.zeros((out_h, out_w, 4), dtype=np.float32)
    heatmap[nodata_mask, 0] = 1.0
    heatmap[nodata_mask, 1] = 0.2
    heatmap[nodata_mask, 2] = 0.2
    heatmap[nodata_mask, 3] = 0.35

    return heatmap, scale


def overlay_grid(
    thumbnail: np.ndarray,
    img_height: int,
    img_width: int,
    tile_size: int,
    overlap: float,
    scale: float,
    skip_ids: Optional[set] = None,
) -> np.ndarray:
    """
    Overlay tile grid lines on a thumbnail.

    Args:
        thumbnail: Thumbnail (H, W, 3) uint8
        img_height: Original raster height
        img_width: Original raster width
        tile_size: Tile size
        overlap: Overlap ratio
        scale: Scale factor
        skip_ids: Tile IDs to mark differently (e.g., already completed)

    Returns:
        Thumbnail with grid overlay (H, W, 3) uint8
    """
    result = thumbnail.copy()
    step = int(tile_size * (1 - overlap))
    h, w = result.shape[:2]

    for y in range(0, img_height, step):
        y_px = int(y * scale)
        if 0 <= y_px < h:
            result[y_px, :, :] = [0, 200, 255]

    for x in range(0, img_width, step):
        x_px = int(x * scale)
        if 0 <= x_px < w:
            result[:, x_px, :] = [0, 200, 255]

    return result


def estimate_crop(config: CropConfig) -> Dict:
    """
    Estimate tiling results without executing the crop.

    Returns:
        {
            "raster_info": RasterInfo,
            "label_info": RasterInfo | None,
            "total_tiles": int,
            "estimated_valid_pct": float,
            "estimated_disk_mb": float,
            "step": int,
            "cols": int,
            "rows": int,
        }
    """
    step = int(config.tile_size * (1 - config.overlap))

    img_info = get_raster_info(config.src_path)
    label_info = get_raster_info(config.label_path) if config.label_path else None

    ref_h = label_info.height if label_info else img_info.height
    ref_w = label_info.width if label_info else img_info.width
    cols = (ref_w + step - 1) // step
    rows = (ref_h + step - 1) // step
    total = cols * rows

    estimated_valid_pct = 0.85

    bytes_per_pixel = np.dtype(img_info.dtype).itemsize * img_info.band_count
    tile_bytes = config.tile_size * config.tile_size * bytes_per_pixel
    compressed_tile_mb = tile_bytes / 2 / (1024 * 1024)
    estimated_disk_mb = total * estimated_valid_pct * compressed_tile_mb

    if not config.single_mode and label_info:
        label_bytes_per_pixel = np.dtype(label_info.dtype).itemsize * label_info.band_count
        label_tile_bytes = config.tile_size * config.tile_size * label_bytes_per_pixel
        estimated_disk_mb += total * estimated_valid_pct * label_tile_bytes / 2 / (1024 * 1024)

    return {
        "raster_info": img_info,
        "label_info": label_info,
        "total_tiles": total,
        "estimated_valid_pct": estimated_valid_pct,
        "estimated_disk_mb": estimated_disk_mb,
        "step": step,
        "cols": cols,
        "rows": rows,
    }
