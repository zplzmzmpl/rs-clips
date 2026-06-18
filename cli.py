"""
cli.py - Command-line entry point

Usage:
    python cli.py --src_path image.tif --output_dir ./output --single_mode --tile_size 1024
    python cli.py --label_path label.tif --src_path image.tif --output_dir ./output --tile_size 1024
    python cli.py --batch_list pairs.txt --output_dir ./output --tile_size 1024
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from core import (
    CropConfig, BatchItem, execute_crop, execute_batch,
    merge_batch_csvs, package_as_dataset, get_raster_info,
)


def parse_batch_file(path: str):
    """
    Parse a batch list file. Each line:
        src_path[,label_path[,suffix]]

    Returns a list of BatchItem.
    """
    items = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 1:
                continue
            src = parts[0]
            label = parts[1] if len(parts) > 1 and parts[1] else None
            suffix = parts[2] if len(parts) > 2 and parts[2] else None
            items.append(BatchItem(src_path=src, label_path=label, suffix=suffix))
    return items


def main():
    parser = argparse.ArgumentParser(
        description="RS-Clips: Remote Sensing Tile Cropper - Single/Dual mode, batch processing, resume support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single mode
  python cli.py --src_path image.tif --output_dir ./clips --single_mode --tile_size 1024

  # Dual mode (label + image)
  python cli.py --label_path label.tif --src_path image.tif --output_dir ./clips --tile_size 1024 --suffix 2010_

  # Resume from checkpoint
  python cli.py --src_path image.tif --output_dir ./clips --single_mode --resume

  # Batch mode (one pair per line: src_path,label_path,suffix)
  python cli.py --batch_list pairs.txt --output_dir ./clips --tile_size 1024

  # With training format packaging
  python cli.py --src_path image.tif --output_dir ./clips --single_mode --package torch
        """
    )

    # Input
    parser.add_argument("--src_path", default=None, help="Source image path")
    parser.add_argument("--label_path", default=None, help="Label image path (required in dual mode)")
    parser.add_argument("--batch_list", default=None,
                        help="Path to batch list file (CSV: src_path[,label_path[,suffix]])")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    # Tiling params
    parser.add_argument("--tile_size", type=int, default=2000, help="Tile size (px), default 2000")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap ratio, default 0.1")
    parser.add_argument("--edge_threshold", type=float, default=0.9,
                        help="Minimum valid fraction for edge tiles, default 0.9")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers, default CPU-1")
    parser.add_argument("--suffix", type=str, default=None, help="Output filename prefix")
    parser.add_argument("--single_mode", action="store_true", help="Enable single-image mode")
    parser.add_argument("--label_nodata", type=str, default=None,
                        help="Label NoData value, e.g. '256' or '256,256,256,256'")
    parser.add_argument("--img_nodata", type=str, default=None,
                        help="Image NoData value, e.g. '0' or '0,0,0'")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Packaging
    parser.add_argument("--package", choices=["torch", "hf"], default=None,
                        help="Auto-package as training dataset (torch=PyTorch, hf=HuggingFace)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio, default 0.8")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio, default 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits, default 42")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Validate
    is_batch = args.batch_list is not None
    if not is_batch and not args.src_path:
        parser.error("--src_path is required when not using --batch_list")
    if not is_batch and not args.single_mode and not args.label_path:
        parser.error("--label_path is required in dual mode (or use --single_mode)")

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    # ── Batch mode ──
    if is_batch:
        items = parse_batch_file(args.batch_list)
        if not items:
            parser.error("Batch list is empty or invalid")

        print(f"Batch mode: {len(items)} image pairs")

        last_print = [0]

        def batch_progress_cb(item_idx, total_items, tile_id, tiles_done, tiles_total):
            now = time.time()
            if now - last_print[0] > 2 or tiles_done == tiles_total:
                overall = (item_idx + tiles_done / max(tiles_total, 1)) / total_items * 100
                print(f"\r  [{overall:5.1f}%] Item {item_idx+1}/{total_items} | "
                      f"{tiles_done}/{tiles_total} tiles | last: {tile_id}", end="", flush=True)
                last_print[0] = now

        results = execute_batch(
            items=items,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            edge_threshold=args.edge_threshold,
            num_workers=args.workers,
            single_mode=args.single_mode,
            label_nodata=args.label_nodata,
            img_nodata=args.img_nodata,
            resume=args.resume,
            progress_callback=batch_progress_cb,
        )

        print(f"\n\n{'='*60}")
        total_valid = 0
        total_time = 0.0
        for item, result in results:
            total_valid += result.valid_tiles
            total_time += result.elapsed_time
            print(f"  {Path(item.src_path).name}: {result.valid_tiles} tiles ({result.elapsed_time:.1f}s)")
        print(f"{'='*60}")
        print(f"Batch total: {total_valid} valid tiles in {total_time:.1f}s")

        # Merge CSVs
        merged_csv = merge_batch_csvs(args.output_dir, args.single_mode)
        print(f"Merged CSV: {merged_csv}")

        # Package
        if args.package:
            pkg_path = package_as_dataset(
                args.output_dir, args.single_mode, format_type=args.package,
                train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed,
            )
            print(f"Dataset packaged ({args.package}): {pkg_path}")

        return

    # ── Single item mode ──
    config = CropConfig(
        src_path=args.src_path,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        edge_threshold=args.edge_threshold,
        num_workers=args.workers,
        suffix=args.suffix,
        single_mode=args.single_mode,
        label_path=args.label_path,
        label_nodata=args.label_nodata,
        img_nodata=args.img_nodata,
        resume=args.resume,
    )

    # Print raster info
    try:
        info = get_raster_info(args.src_path)
        print(f"Source: {info.width}x{info.height} px | {info.band_count} bands | {info.dtype}")
        if not args.single_mode and args.label_path:
            linfo = get_raster_info(args.label_path)
            print(f"Label:  {linfo.width}x{linfo.height} px | {linfo.band_count} bands | {linfo.dtype}")
    except Exception as e:
        print(f"Warning: cannot read raster info: {e}")

    last_print = [0]

    def progress_cb(completed, total, tile_id):
        pct = completed / total * 100 if total > 0 else 0
        now = time.time()
        if now - last_print[0] > 2 or completed == total:
            print(f"\r  [{pct:5.1f}%] {completed}/{total} tiles | last: {tile_id}", end="", flush=True)
            last_print[0] = now

    print(f"\nStarting: mode={'single' if args.single_mode else 'dual'}, "
          f"tile={args.tile_size}px, overlap={args.overlap:.0%}, "
          f"workers={args.workers or 'auto'}\n")

    result = execute_crop(config, progress_callback=progress_cb)

    print(f"\n\n{'='*50}")
    print(f"Done")
    print(f"  Valid tiles:  {result.valid_tiles}")
    print(f"  NoData skip:  {result.skipped_nodata}")
    print(f"  Too small:    {result.skipped_small}")
    print(f"  Resumed:      {result.skipped_existing}")
    print(f"  Errors:       {result.errors}")
    print(f"  Time:         {result.elapsed_time:.1f}s")
    print(f"  CSV:          {result.csv_path}")
    print(f"{'='*50}")

    # Package
    if args.package:
        pkg_path = package_as_dataset(
            args.output_dir, args.single_mode, format_type=args.package,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed,
        )
        print(f"Dataset packaged ({args.package}): {pkg_path}")


if __name__ == "__main__":
    main()
