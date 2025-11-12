# SAM 2 Mask Generator
A comprehensive software package designed for automatic Digital Image Correlation (DIC) or other Region of Interest (ROI) recognition, mask generation, and smoothing. This tool leverages the Segment Anything Model 2 (SAM 2) for intelligent mask generation and provides advanced spatial and temporal smoothing capabilities.

## Overview and Demo videos

Welcome to watch our code's demonstration videos! This software package consists of two primary components:

- **SAM 2 Mask Generator** (Python-based GUI application)
![Demo video](assets/Mask_generator_demo_video.gif)

Or you can access the above demo video via this link: https://www.youtube.com/watch?v=hMHhlv2D3rA (youtube)

- **Mask Smoothing Module** (MATLAB-based processing tools)
![Demo video](assets/Smoothing_demo_video.gif)

Or you can access the above demo video via this link: [https://www.youtube.com/watch?v=hMHhlv2D3rA](https://www.youtube.com/watch?v=VNcFvScoI-E) (youtube)

The complete source code, along with installation instructions, is available in our GitHub repository. Users should consult this README.md file for detailed installation and setup instructions to ensure proper configuration for use.


## Features

### SAM 2 Mask Generator
- **Interactive GUI**: User-friendly interface for mask generation
- **Multi-model Support**: Four pre-trained SAM 2 models (Large, Base Plus, Small, Tiny)
- **Real-time Processing**: Live mask display and progress tracking
- **GPU/CPU Support**: Flexible computing resource utilization
- **Batch Processing**: Efficient handling of image sequences
- **Point-based Annotation**: Interactive foreground/background point marking

### Mask Smoothing Module
- **Spatial Smoothing**: Advanced diffusion-based smoothing algorithms
- **Temporal Smoothing**: Frame-to-frame consistency enhancement
- **Real-time Visualization**: Live preview of smoothing progress
- **Parameter Optimization**: Comprehensive control over smoothing parameters
- **Noise Reduction**: Intelligent handling of high-noise frames

## Installation

### Prerequisites
- Python 3.8 or higher
- MATLAB (for Mask Smoothing Module)
- CUDA-compatible GPU (optional, for accelerated processing)

### Python Dependencies
## Method 1:
```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install pillow
pip install numpy
pip install tqdm
pip install hydra-core
pip install iopath
pip install tkinter
```
## Method 2:
```bash
pip install -e .
```

### SAM 2 Model Download
Download the pre-trained SAM 2 models from the official repository and place them in the `checkpoints/` directory:
- `sam2.1_hiera_large.pt`
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_tiny.pt`

## Usage

### SAM 2 Mask Generator

1. **Launch the Application**
   ```bash
   python main.py
   ```

2. **Configure Input/Output**
   - Select input directory containing raw DIC images
   - Choose output directory for generated masks
   - Set frame range for processing (optional)

3. **Mark Control Points**
   - Click "Start Drawing Points" to enable annotation mode
   - Mark foreground points (red) for target ROI
   - Mark background points (blue) for excluded regions
   - Use "Mode" toggle to switch between foreground/background

4. **Select Processing Options**
   - Choose computing device (GPU/CPU)
   - Select SAM 2 model size based on accuracy/speed requirements
   - Configure processing parameters

5. **Generate Masks**
   - Click "Start Processing" to begin mask generation
   - Monitor real-time progress and results
   - Generated masks are automatically saved to output directory

### Mask Smoothing Module

#### Spatial Smoothing
1. Launch `imageSmoothingGUI.m` in MATLAB
2. Select "Spatial Smoothing" mode
3. Configure input/output directories
4. Adjust smoothing parameters:
   - **Iteration**: Maximum smoothing iterations
   - **Time Step**: Smoothing intensity per iteration
   - **Lambda**: Gradient sensitivity control
   - **Gaussian Sigma/Size**: Filter spread and window size
5. Enable "Real-time Visualization" for live preview
6. Start processing

#### Temporal Smoothing
1. Select "Temporal Smoothing" mode
2. Set input directory to spatially smoothed masks
3. Configure temporal parameters:
   - **Variance Threshold**: Maximum allowable variance for noise detection
   - **Number of Neighbors**: Adjacent frames for continuity
   - **Gaussian Sigma**: Temporal filter spread
   - **3D Kernel Size**: Spatial-temporal smoothing extent
4. Process the sequence for temporal consistency

## Parameters Guide

### SAM 2 Models
- **Large**: Highest accuracy, slower processing
- **Base Plus**: Balanced accuracy and speed
- **Small**: Faster processing, moderate accuracy
- **Tiny**: Fastest processing, basic accuracy

### Spatial Smoothing Parameters
- **Iteration**: Controls smoothing intensity (typically 10-50)
- **Time Step**: Affects convergence speed (0.1-1.0)
- **Lambda**: Gradient sensitivity (0.1-10.0)
- **Gaussian Sigma**: Filter spread (1.0-5.0)
- **Gaussian Size**: Window size (3-15)

### Temporal Smoothing Parameters
- **Variance Threshold**: Noise detection sensitivity (0.01-0.1)
- **Neighbors**: Temporal window size (3-7)
- **Gaussian Sigma**: Temporal smoothing (0.5-2.0)
- **3D Kernel Size**: Combined smoothing (3-7)

## File Structure
    Mask_generater/
    ├── main.py # SAM 2 Mask Generator GUI
    ├── imageSmoothingGUI.m # Mask Smoothing Module
    ├── checkpoints/ # SAM 2 model files
    ├── examples/ # Sample images and masks
    ├── sam2/ # SAM 2 implementation
    ├── training/ # Training utilities
    └── tools/ # Additional tools


## Performance Tips

1. **Model Selection**: Choose appropriate model size based on accuracy requirements
2. **Batch Processing**: Process large sequences in batches
3. **Memory Management**: Monitor system memory during large-scale processing
4. **Parameter Tuning**: Experiment with smoothing parameters for optimal results

## Troubleshooting

### Common Issues
- **CUDA Out of Memory**: Reduce batch size or use smaller model
- **Slow Processing**: Switch to GPU or use smaller model
- **Poor Mask Quality**: Add more control points or adjust smoothing parameters
- **MATLAB Errors**: Ensure MATLAB is properly installed and licensed

## Support and Contact
For technical support and bug reports, please refer to the GitHub repository issues page.

**Contact:**
- Zixiang (Zach) Tong: zachtong@utexas.edu
- Jin Yang: jin.yang@austin.utexas.edu
- For other ALDIC/DVC questions/support, please email to: aldicdvc@gmail.com

## Citation

If you use this software in your research, please cite our preprint:
https://www.researchsquare.com/article/rs-5566473/v1
