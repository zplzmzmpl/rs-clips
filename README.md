# RS-Clips

A remote sensing image tiling tool for deep learning — crop large GeoTIFF rasters into training-ready tile datasets.

## Features

- **Single & Dual mode** — crop one image, or synchronize label+image pairs with automatic geospatial alignment
- **Batch processing** — tile multiple image pairs in one run
- **Visual preview** — thumbnail with grid overlay, NoData heatmap, disk usage estimate
- **Resume support** — checkpoint every 50 tiles; restart from where you left off
- **Training format packaging** — auto-generate PyTorch `Dataset` loader or HuggingFace dataset structure with train/val/test splits
- **Full parameter control** — tile size, overlap, edge tolerance, NoData values, workers, all configurable
- **Zero heavy UI dependencies** — Streamlit frontend (browser-based) + CLI for scripting

## Quick Start

### Install

```bash
git clone https://github.com/zplzmzmpl/rs-clips.git
cd rs-clips
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Web UI (recommended)

```bash
streamlit run app.py
```

Open your browser, configure parameters visually, preview the grid, and click **Start Cropping**.

### Command Line

```bash
# Single image mode
python cli.py --src_path image.tif --output_dir ./clips --single_mode --tile_size 1024

# Dual mode (label + image synchronized)
python cli.py --label_path label.tif --src_path image.tif --output_dir ./clips --tile_size 1024

# Batch mode (one pair per line in a CSV file)
python cli.py --batch_list pairs.txt --output_dir ./clips --tile_size 1024

# With resume and training packaging
python cli.py --src_path image.tif --output_dir ./clips --single_mode --resume --package torch
```

**Batch list format** (`pairs.txt`):
```
/path/to/image1.tif,/path/to/label1.tif,2010_
/path/to/image2.tif,/path/to/label2.tif,2015_
/path/to/image3.tif   # single mode if no label
```

## Architecture

```
rs-clips/
├── core.py        # Core logic (no UI dependency, callable from any Python code)
├── preview.py     # Thumbnail, grid overlay, NoData heatmap generation
├── app.py         # Streamlit web frontend
├── cli.py         # Command-line interface
└── requirements.txt
```

The core layer (`core.py`) is completely decoupled from UI — you can import and call it directly:

```python
from core import CropConfig, execute_crop

config = CropConfig(
    src_path="image.tif",
    output_dir="./output",
    tile_size=1024,
    overlap=0.1,
    single_mode=True,
)
result = execute_crop(config)
print(f"Generated {result.valid_tiles} tiles in {result.elapsed_time:.1f}s")
```

## Training Format Packaging

After tiling, optionally auto-package into a training-ready format:

### PyTorch

```bash
python cli.py --src_path image.tif --output_dir ./clips --single_mode --package torch
```

Generates:
```
output/
├── dataset_index/
│   ├── train.txt          # File list: image_path [label_path]
│   ├── val.txt
│   ├── test.txt
│   └── tile_dataset.py    # Ready-to-use PyTorch Dataset class
```

Usage:
```python
from tile_dataset import TileDataset
from torch.utils.data import DataLoader

train_ds = TileDataset("./output", split="train")
loader = DataLoader(train_ds, batch_size=8, shuffle=True)
```

### HuggingFace

```bash
python cli.py --src_path image.tif --output_dir ./clips --single_mode --package hf
```

Generates:
```
output/
├── hf_dataset/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── README.md          # Dataset card with metadata
```

## Key Improvements Over the Original Script

| Aspect | Before | After |
|--------|--------|-------|
| Source file I/O | Opens source raster twice per tile | Single open per tile |
| Edge padding | Python for-loop, row by row | `np.pad(mode='edge')` — one call |
| Edge threshold | Hardcoded 0.9 | Configurable via UI/CLI |
| Visualization | None | Thumbnail + grid + NoData heatmap |
| Pre-run estimate | None | Tile count, disk usage, grid layout |
| Resume | None | Checkpoint every 50 tiles |
| Architecture | Logic/UI/CLI coupled | Layered: core ↔ preview ↔ app/cli |
| Progress | tqdm text only | Streamlit progress bar / CLI real-time % |
| NoData insight | Print only | NoData distribution heatmap + statistics |
| Batch processing | Not supported | Multiple image pairs in one run |
| Training packaging | Not supported | PyTorch Dataset / HuggingFace JSONL |

## Parameters Reference

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Source image | `--src_path` | required | Path to source raster |
| Label image | `--label_path` | None | Path to label raster (dual mode) |
| Output dir | `--output_dir` | required | Output directory |
| Tile size | `--tile_size` | 2000 | Square tile edge length (px) |
| Overlap | `--overlap` | 0.1 | Overlap fraction between tiles |
| Edge threshold | `--edge_threshold` | 0.9 | Min valid fraction for edge tiles |
| Workers | `--workers` | CPU-1 | Parallel processes |
| Suffix | `--suffix` | None | Output filename prefix |
| Single mode | `--single_mode` | False | Single-image mode |
| Label NoData | `--label_nodata` | auto | Custom NoData for labels |
| Image NoData | `--img_nodata` | auto | Custom NoData for images |
| Resume | `--resume` | False | Resume from checkpoint |
| Package | `--package` | None | `torch` or `hf` |
| Train ratio | `--train_ratio` | 0.8 | Train split ratio |
| Val ratio | `--val_ratio` | 0.1 | Val split ratio |
| Seed | `--seed` | 42 | Random seed for splits |

## License

MIT
