"""
app.py - Streamlit frontend for rs-clips

Usage:
    streamlit run app.py

Features:
    - File selection / parameter configuration
    - Batch processing support
    - Thumbnail + grid preview
    - NoData area visualization
    - Real-time progress
    - Training format packaging option
    - Tile result sampling preview
"""

import streamlit as st
from pathlib import Path
import numpy as np
import os
import logging

from core import (
    CropConfig, BatchItem, execute_crop, execute_batch,
    merge_batch_csvs, package_as_dataset,
    get_raster_info, parse_nodata, load_checkpoint,
)
from preview import (
    generate_thumbnail, generate_nodata_heatmap, overlay_grid, estimate_crop,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="RS-Clips: Remote Sensing Tile Cropper",
    page_icon="satellite",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def format_size(mb: float) -> str:
    if mb < 1:
        return f"{mb * 1024:.0f} KB"
    elif mb < 1024:
        return f"{mb:.1f} MB"
    else:
        return f"{mb / 1024:.1f} GB"


def raster_info_table(info) -> None:
    """Display raster metadata table."""
    st.markdown(f"""
    | Property | Value |
    |----------|-------|
    | Dimensions | {info.width} x {info.height} px |
    | Bands | {info.band_count} |
    | Data type | {info.dtype} |
    | CRS | {info.crs or 'Unknown'} |
    | Resolution | {info.resolution} |
    | NoData | {info.nodata} |
    | Built-in overviews | {'Yes' if info.has_overviews else 'No'} |
    """)


# ──────────────────────────────────────────────
# Main interface
# ──────────────────────────────────────────────

st.title("RS-Clips: Remote Sensing Tile Cropper")
st.caption("Large remote sensing images -> deep learning training tiles | Single/Dual mode, batch processing, resume support")

# ══════════════════════════════════════════════
# Sidebar - Parameters
# ══════════════════════════════════════════════

with st.sidebar:
    st.header("Data Source")

    mode = st.radio(
        "Crop mode",
        ["Single image", "Image + Label (synchronized)"],
        index=0,
        help="Single: crop one image; Dual: synchronized label+image pair"
    )
    single_mode = mode == "Single image"

    processing = st.radio(
        "Processing",
        ["Single image pair", "Batch (multiple pairs)"],
        index=0,
        help="Batch mode processes multiple image pairs in sequence"
    )
    is_batch = processing == "Batch (multiple pairs)"

    if is_batch:
        st.subheader("Batch Input")
        src_paths_text = st.text_area(
            "Source image paths (one per line)",
            placeholder="/path/to/image1.tif\n/path/to/image2.tif",
            height=100,
        )
        src_paths = [p.strip() for p in src_paths_text.strip().splitlines() if p.strip()]

        label_paths = []
        suffixes = []
        if not single_mode:
            label_paths_text = st.text_area(
                "Label image paths (one per line, same order)",
                placeholder="/path/to/label1.tif\n/path/to/label2.tif",
                height=100,
            )
            label_paths = [p.strip() for p in label_paths_text.strip().splitlines() if p.strip()]

        suffixes_text = st.text_area(
            "File suffixes (one per line, optional)",
            placeholder="2010_\n2015_",
            height=80,
        )
        suffixes = [p.strip() if p.strip() else None for p in suffixes_text.strip().splitlines()]
    else:
        src_path = st.text_input(
            "Source image path",
            placeholder="/path/to/image.tif",
        )
        label_path = None
        if not single_mode:
            label_path = st.text_input(
                "Label image path",
                placeholder="/path/to/label.tif",
            )

    output_dir = st.text_input(
        "Output directory",
        placeholder="/path/to/output",
    )

    st.divider()
    st.header("Tiling Parameters")

    tile_size = st.slider(
        "Tile size (px)",
        min_value=128, max_value=8192, value=1024, step=128,
        help="Square tile edge length in pixels"
    )

    overlap = st.slider(
        "Overlap ratio",
        min_value=0.0, max_value=0.5, value=0.1, step=0.05,
        help="Overlap fraction between adjacent tiles"
    )

    edge_threshold = st.slider(
        "Edge tolerance",
        min_value=0.5, max_value=1.0, value=0.9, step=0.05,
        help="Minimum valid fraction for edge tiles; tiles below this are skipped"
    )

    suffix_input = st.text_input(
        "File prefix (single mode)",
        value="",
        help="Output filename prefix, e.g. '2010_'",
    ) if not is_batch else None

    col1, col2 = st.columns(2)
    with col1:
        num_workers = st.number_input(
            "Workers",
            min_value=1, max_value=64,
            value=max(1, os.cpu_count() - 1),
            help="Parallel processes (recommended: CPU cores - 1)"
        )
    with col2:
        resume = st.checkbox(
            "Resume",
            value=False,
            help="Resume from last checkpoint; skip already completed tiles"
        )

    st.divider()
    st.header("NoData Settings")

    nodata_mode_label = st.radio(
        "NoData source",
        ["Auto-detect (from raster metadata)", "Manual"],
        index=0,
    )
    manual_nodata = nodata_mode_label == "Manual"

    img_nodata_str = None
    label_nodata_str = None

    if manual_nodata:
        img_nodata_str = st.text_input(
            "Image NoData",
            placeholder="0,0,0",
            help="Comma-separated, e.g. 0,0,0 or 256"
        )
        if not single_mode:
            label_nodata_str = st.text_input(
                "Label NoData",
                placeholder="256,256,256,256",
                help="Comma-separated"
            )

    st.divider()
    st.header("Output Packaging")

    package_enabled = st.checkbox(
        "Auto-package as training dataset",
        value=False,
        help="After tiling, generate train/val/test splits and a ready-to-use Dataset loader"
    )

    if package_enabled:
        format_type = st.selectbox(
            "Dataset format",
            ["PyTorch", "HuggingFace"],
            index=0,
            help="PyTorch: file lists + Dataset class; HuggingFace: JSONL + dataset card"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            train_ratio = st.number_input("Train %", value=80, min_value=10, max_value=90) / 100
        with c2:
            val_ratio = st.number_input("Val %", value=10, min_value=5, max_value=40) / 100
        with c3:
            seed = st.number_input("Seed", value=42, min_value=0)

# ══════════════════════════════════════════════
# Main area
# ══════════════════════════════════════════════

# Build config for preview (single mode preview only)
if not is_batch:
    config = CropConfig(
        src_path=src_path or "",
        output_dir=output_dir or "",
        tile_size=tile_size,
        overlap=overlap,
        edge_threshold=edge_threshold,
        num_workers=int(num_workers),
        suffix=suffix_input if suffix_input else None,
        single_mode=single_mode,
        label_path=label_path,
        label_nodata=label_nodata_str if manual_nodata else None,
        img_nodata=img_nodata_str if manual_nodata else None,
        resume=resume,
    )

# ── Info + Preview tabs ──

tab_info, tab_preview, tab_nodata = st.tabs(["Raster Info", "Grid Preview", "NoData Map"])

# --- Info Tab ---
with tab_info:
    if not is_batch and src_path and os.path.isfile(src_path):
        try:
            img_info = get_raster_info(src_path)
            st.subheader("Source Image")
            raster_info_table(img_info)
        except Exception as e:
            st.error(f"Cannot read source image: {e}")

        if not single_mode and label_path and os.path.isfile(label_path):
            try:
                lbl_info = get_raster_info(label_path)
                st.subheader("Label Image")
                raster_info_table(lbl_info)
            except Exception as e:
                st.error(f"Cannot read label image: {e}")

        try:
            est = estimate_crop(config)
            st.subheader("Tiling Estimate")
            c1, c2, c3 = st.columns(3)
            c1.metric("Grid", f"{est['cols']} x {est['rows']}")
            c2.metric("Total positions", f"{est['total_tiles']}")
            c3.metric("Est. disk usage", format_size(est['estimated_disk_mb']))
        except Exception as e:
            st.warning(f"Estimate failed: {e}")
    elif is_batch:
        st.info("Batch mode: enter image paths in the sidebar, then click Start. Preview is shown per-item during processing.")
    else:
        st.info("Enter an image path in the sidebar to view raster info")

# --- Preview Tab ---
with tab_preview:
    if not is_batch and src_path and os.path.isfile(src_path):
        preview_path = label_path if (not single_mode and label_path and os.path.isfile(label_path)) else src_path

        with st.spinner("Generating thumbnail..."):
            try:
                thumb_nodata = None
                if manual_nodata and img_nodata_str:
                    with rasterio.open(src_path) as tmp:
                        thumb_nodata = parse_nodata(img_nodata_str, tmp.count)

                thumbnail, scale = generate_thumbnail(preview_path, nodata=thumb_nodata)
                ref_info = get_raster_info(preview_path)
                grid_thumb = overlay_grid(
                    thumbnail,
                    img_height=ref_info.height,
                    img_width=ref_info.width,
                    tile_size=tile_size,
                    overlap=overlap,
                    scale=scale,
                )

                st.image(grid_thumb, caption="Tile grid preview (cyan lines = tile boundaries)", use_container_width=True)

                if resume and output_dir:
                    existing = load_checkpoint(output_dir, config.params_hash())
                    if existing:
                        st.info(f"Resume: {len(existing)} completed tiles found")
                    else:
                        st.info("Resume mode enabled, no checkpoint found, starting from scratch")

            except Exception as e:
                st.error(f"Preview generation failed: {e}")
    elif is_batch:
        st.info("Grid preview is generated per-item during batch processing")
    else:
        st.info("Enter an image path to view grid preview")

# --- NoData Tab ---
with tab_nodata:
    if not is_batch and src_path and os.path.isfile(src_path):
        with st.spinner("Analyzing NoData distribution..."):
            try:
                nodata_val = None
                if manual_nodata and img_nodata_str:
                    with rasterio.open(src_path) as tmp:
                        nodata_val = parse_nodata(img_nodata_str, tmp.count)

                heatmap, scale = generate_nodata_heatmap(src_path, nodata=nodata_val)
                thumbnail, _ = generate_thumbnail(src_path, nodata=nodata_val)

                composite = thumbnail.astype(np.float32) / 255.0
                for c in range(3):
                    composite[:, :, c] = composite[:, :, c] * (1 - heatmap[:, :, 3]) + heatmap[:, :, c] * heatmap[:, :, 3]
                composite = (composite * 255).clip(0, 255).astype(np.uint8)

                st.image(composite, caption="NoData distribution (red areas = NoData pixels)", use_container_width=True)

                nodata_pct = np.sum(heatmap[:, :, 3] > 0) / heatmap.shape[0] / heatmap.shape[1] * 100
                if nodata_pct > 0:
                    st.warning(f"NoData area: {nodata_pct:.1f}% - tiles in these regions will be skipped")
                else:
                    st.success("No NoData areas detected (or NoData value not specified)")
            except Exception as e:
                st.error(f"NoData analysis failed: {e}")
    elif is_batch:
        st.info("NoData map is generated per-item during batch processing")
    else:
        st.info("Enter an image path to view NoData distribution")

# ══════════════════════════════════════════════
# Execute
# ══════════════════════════════════════════════

st.divider()

# Validate inputs
if is_batch:
    can_run = bool(src_paths and output_dir)
    if not single_mode:
        can_run = can_run and len(label_paths) == len(src_paths)
else:
    can_run = bool(src_path and output_dir)
    if not single_mode:
        can_run = can_run and bool(label_path)

if not can_run:
    st.warning("Please fill in the required paths before starting")
else:
    c1, c2 = st.columns([1, 4])
    with c1:
        run_btn = st.button("Start Cropping", type="primary", use_container_width=True)
    with c2:
        extras = []
        if resume:
            extras.append("Resume enabled")
        if package_enabled:
            extras.append(f"Auto-package ({format_type})")
        if extras:
            st.info(" | ".join(extras))

    if run_btn:
        # ── Single item ──
        if not is_batch:
            if not os.path.isfile(src_path):
                st.error(f"Source image not found: {src_path}")
                st.stop()
            if not single_mode and label_path and not os.path.isfile(label_path):
                st.error(f"Label image not found: {label_path}")
                st.stop()

            progress_bar = st.progress(0, text="Preparing...")
            status_container = st.empty()

            def progress_cb(completed, total, tile_id):
                pct = completed / total if total > 0 else 0
                progress_bar.progress(pct, text=f"Tiling... {completed}/{total} ({pct:.0%})")

            try:
                result = execute_crop(config, progress_callback=progress_cb)
                progress_bar.progress(1.0, text="Tiling complete!")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Valid tiles", f"{result.valid_tiles}")
                c2.metric("NoData skipped", f"{result.skipped_nodata}")
                c3.metric("Too small", f"{result.skipped_small}")
                c4.metric("Time", f"{result.elapsed_time:.1f}s")

                if result.errors > 0:
                    st.warning(f"{result.errors} tiles failed - check logs")

                st.success(f"CSV saved to: `{result.csv_path}`")

                # Package if enabled
                if package_enabled:
                    fmt = "torch" if format_type == "PyTorch" else "hf"
                    pkg_path = package_as_dataset(
                        output_dir, single_mode, format_type=fmt,
                        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
                    )
                    st.success(f"Dataset packaged to: `{pkg_path}`")

            except Exception as e:
                progress_bar.empty()
                st.error(f"Tiling failed: {e}")
                if resume:
                    st.info("Resume is enabled - click Start again to continue from checkpoint")

        # ── Batch ──
        else:
            # Build batch items
            items = []
            for i, sp in enumerate(src_paths):
                lp = label_paths[i] if i < len(label_paths) else None
                sf = suffixes[i] if i < len(suffixes) else None
                items.append(BatchItem(src_path=sp, label_path=lp, suffix=sf))

            progress_bar = st.progress(0, text="Preparing batch...")
            status_text = st.empty()

            def batch_progress_cb(item_idx, total_items, tile_id, tiles_done, tiles_total):
                overall = (item_idx + tiles_done / max(tiles_total, 1)) / total_items
                progress_bar.progress(overall, text=f"Batch [{item_idx+1}/{total_items}] {tiles_done}/{tiles_total} tiles")
                status_text.text(f"Processing item {item_idx+1}/{total_items}: {Path(src_paths[item_idx]).name} | last tile: {tile_id}")

            try:
                results = execute_batch(
                    items=items,
                    output_dir=output_dir,
                    tile_size=tile_size,
                    overlap=overlap,
                    edge_threshold=edge_threshold,
                    num_workers=int(num_workers),
                    single_mode=single_mode,
                    label_nodata=label_nodata_str if manual_nodata else None,
                    img_nodata=img_nodata_str if manual_nodata else None,
                    resume=resume,
                    progress_callback=batch_progress_cb,
                )

                progress_bar.progress(1.0, text="Batch complete!")
                st.subheader("Batch Results")

                total_valid = 0
                total_time = 0.0
                for item, result in results:
                    total_valid += result.valid_tiles
                    total_time += result.elapsed_time
                    with st.expander(f"{Path(item.src_path).name} - {result.valid_tiles} tiles ({result.elapsed_time:.1f}s)"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Valid", result.valid_tiles)
                        c2.metric("Skipped (NoData)", result.skipped_nodata)
                        c3.metric("Errors", result.errors)

                st.success(f"Batch total: {total_valid} valid tiles in {total_time:.1f}s")

                # Merge CSVs
                merged_csv = merge_batch_csvs(output_dir, single_mode)
                st.success(f"Merged CSV: `{merged_csv}`")

                # Package if enabled
                if package_enabled:
                    fmt = "torch" if format_type == "PyTorch" else "hf"
                    pkg_path = package_as_dataset(
                        output_dir, single_mode, format_type=fmt,
                        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
                    )
                    st.success(f"Dataset packaged to: `{pkg_path}`")

            except Exception as e:
                progress_bar.empty()
                st.error(f"Batch failed: {e}")

# ── Result sampling preview ──
if output_dir and os.path.isdir(output_dir):
    # Check for batch subdirectories or single-item output
    preview_dirs = []
    if is_batch:
        for entry in sorted(os.listdir(output_dir)):
            sub = os.path.join(output_dir, entry)
            if os.path.isdir(sub):
                preview_dirs.append((entry, sub))
    else:
        preview_dirs = [("", output_dir)]

    for label, pdir in preview_dirs:
        images_dir = os.path.join(pdir, "images") if single_mode else os.path.join(pdir, "HR_img")
        if os.path.isdir(images_dir):
            tif_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
            if tif_files:
                display_label = f" ({label})" if label else ""
                with st.expander(f"Tile sample preview{display_label} ({len(tif_files)} tiles total)", expanded=False):
                    sample_n = min(6, len(tif_files))
                    sample_indices = np.random.choice(len(tif_files), sample_n, replace=False)
                    sample_indices.sort()

                    cols = st.columns(3)
                    for i, idx in enumerate(sample_indices):
                        fn = tif_files[idx]
                        fpath = os.path.join(images_dir, fn)
                        try:
                            thumb, _ = generate_thumbnail(fpath, max_size=200)
                            cols[i % 3].image(thumb, caption=fn, use_container_width=True)
                        except Exception:
                            cols[i % 3].text(fn)
