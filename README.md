# SAM2 Studio — Mask Generator for DIC & ROI Recognition

A professional GUI application for automatic Digital Image Correlation (DIC) and Region of Interest (ROI) mask generation using Segment Anything Model 2 (SAM2). Built with PyQt6 and MVC architecture.

## Features

- **Interactive Annotation**: Point-based foreground/background marking with undo/redo
- **Multi-model Support**: SAM2 Hiera Large / Base Plus / Small / Tiny
- **GPU Accelerated**: CUDA auto-detection for RTX/Tesla GPUs (~100x faster than CPU)
- **Real-time Preview**: Live mask overlay during propagation
- **Smart Processing**: Auto-downsamples large images for inference, upscales masks to original resolution
- **Post-processing**: Built-in Perona-Malik spatial smoothing and 3D Gaussian temporal smoothing
- **Mid-sequence Correction**: Re-propagate from any frame with new annotation points
- **Keyboard Shortcuts**: Full keyboard workflow (V/D/E/Space/Ctrl+Z/Y/S/O)

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- SAM2 model checkpoints

### Step 1: Install PyTorch with CUDA

Check your NVIDIA driver version with `nvidia-smi`, then install the matching PyTorch:

```bash
# CUDA 12.8 (driver >= 570)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6 (driver >= 560)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU only (no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify CUDA is working:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX ...
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### Step 3: Download SAM2 checkpoints

Download model files from [facebookresearch/sam2](https://github.com/facebookresearch/sam2) and place in `checkpoints/`:

```
checkpoints/
├── sam2.1_hiera_large.pt
├── sam2.1_hiera_base_plus.pt
├── sam2.1_hiera_small.pt
└── sam2.1_hiera_tiny.pt
```

## Usage

```bash
python main.py
```

### Workflow

1. **Set directories** — Paste or browse input image folder and output folder
2. **Navigate frames** — Use slider or Left/Right arrow keys
3. **Annotate** — Click to add foreground points (V), background points (D); Ctrl+Z to undo
4. **Process** — Click Start Processing or press Ctrl+Enter
5. **Review** — Browse generated masks with real-time overlay
6. **Correct** — Add correction points on any frame, re-propagate (E to enter correction mode)
7. **Smooth** — Apply spatial or temporal smoothing from the sidebar

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| V | Foreground point mode |
| D | Background point mode |
| E | Enter correction mode |
| Space | Toggle overlay visibility |
| Ctrl+Z | Undo annotation |
| Ctrl+Y | Redo annotation |
| Ctrl+S | Save annotation config |
| Ctrl+O | Load annotation config |
| Ctrl+Enter | Start processing |
| Left/Right | Previous/Next frame |
| Home/End | First/Last frame |
| Escape | Cancel processing |

## Project Structure

```
Mask_generater/
├── main.py                    # Application entry point
├── controllers/               # MVC controllers
│   ├── app_state.py           # Central state manager
│   ├── annotation_controller.py  # Undo/redo annotation commands
│   ├── processing_controller.py  # Background mask generation workers
│   └── smoothing_controller.py   # Spatial/temporal smoothing workers
├── core/                      # Core algorithms
│   ├── mask_generator.py      # SAM2 video predictor wrapper
│   ├── image_processing.py    # Image loading/conversion utilities
│   ├── spatial_smoothing.py   # Perona-Malik anisotropic diffusion
│   └── temporal_smoothing.py  # 3D Gaussian temporal smoothing
├── gui/                       # PyQt6 GUI
│   ├── main_window.py         # Main window with signal wiring
│   ├── panels/                # Sidebar, canvas, frame navigator, status bar
│   ├── widgets/               # Reusable widgets (path selector, etc.)
│   ├── theme.py               # Dark theme colors and fonts
│   └── icons.py               # SVG icon generator
├── utils/                     # Utilities
│   └── device_manager.py      # GPU detection and VRAM monitoring
├── sam2/                      # SAM2 model implementation (Meta)
├── checkpoints/               # Model weight files (not tracked)
├── tests/                     # Test suite (172 tests)
└── requirements.txt           # Python dependencies
```

## Performance Tips

- **Use GPU**: CUDA gives ~100x speedup over CPU for mask propagation
- **Model selection**: `hiera_tiny` is ~5x faster than `hiera_large` with acceptable quality for most DIC
- **Auto-downsampling**: Large images (>1024px) are automatically downsampled for inference, masks are upscaled to original resolution
- **Parallel conversion**: Image format conversion uses multi-threaded I/O

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `torch.cuda.is_available()` returns `False` | You have CPU-only PyTorch. Reinstall with CUDA index URL (see Step 1) |
| CUDA out of memory | Use a smaller model (Tiny/Small) or reduce frame count |
| Slow processing | Check GPU is detected (shown in status bar). Use Tiny model for preview |
| `Cannot find primary config` | Ensure `sam2/configs/` directory exists with yaml files |
| Masks look blocky | Expected from auto-downsampling. Adjust annotation points for better coverage |

## Citation

If you use this software in your research, please cite:
https://www.researchsquare.com/article/rs-5566473/v1
