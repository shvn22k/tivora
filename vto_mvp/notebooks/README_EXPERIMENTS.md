# Tivora VTO Performance Experiments Guide

## Overview

This notebook (`experiments-03.ipynb`) systematically tests different Virtual Try-On approaches to determine the fastest viable solution for processing 50-100 garments in under 10-30 seconds.

## Quick Start

### 1. Prerequisites

Make sure you have:
- CUDA-enabled GPU (8GB+ VRAM recommended)
- Python 3.10+
- Jupyter Notebook or JupyterLab

### 2. Install Dependencies

Run these commands before running the notebook:

```bash
# PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Diffusion models and tools
pip install diffusers transformers accelerate
pip install opencv-python pillow numpy matplotlib pandas
pip install controlnet-aux
pip install timm safetensors huggingface-hub

# SAM 2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Optional but recommended for speed
pip install xformers
```

### 3. Download SAM 2.1 Model (Optional)

```bash
cd vto_mvp/notebooks
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### 4. Prepare Test Images (Optional)

Place your test images in the `vto_mvp/notebooks/` directory:
- `person.jpg` - Full body person image (facing forward)
- `garment.jpg` - Garment product photo

If you don't provide images, the notebook will use placeholders.

### 5. Run the Notebook

1. Open `experiments-03.ipynb` in Jupyter
2. Run cells sequentially from top to bottom
3. Each experiment will output timing results and visualizations
4. Check the Final Summary at the end for comprehensive comparison

## What Each Experiment Does

### Experiment 1: Environment Check
- Verifies GPU availability
- Reports VRAM and GPU name
- Creates helper functions

### Experiment 2: Load Test Images
- Loads and resizes person/garment images to 512x768
- Times the loading operations

### Experiment 3: SAM 2.1 Person Segmentation
- Tests person segmentation speed
- Runs 10 iterations for accurate timing
- Visualizes segmentation masks

### Experiment 4: RMBG-1.4 Garment Segmentation
- Tests background removal from garments
- Measures segmentation speed
- Shows clean garment images

### Experiment 5: OpenPose Extraction
- Extracts pose keypoints from person image
- Times the pose detection
- Visualizes pose skeleton

### Experiment 6: SD 1.5 Baseline
- Tests standard Stable Diffusion 1.5 with ControlNet
- Compares 10, 20, and 30 inference steps
- Establishes slow baseline for comparison

### Experiment 7: LCM Fast Approach
- Tests Latent Consistency Model with SDXL
- Tests 1, 2, 4, and 8 steps
- Shows quality vs speed trade-offs

### Experiment 8: Batch Processing
- Tests sequential vs batched inference
- Finds optimal batch size for GPU
- Measures throughput improvements

### Experiment 9: Full Pipeline (50 Garments)
- Simulates realistic catalog browsing
- Includes preprocessing + generation
- Projects timing for 100 garments

### Experiment 10: Full Pipeline (100 Garments)
- Tests actual target use case
- Determines if <10s or <30s goals are met
- Provides optimization recommendations

### Experiment 11: Speed Optimizations
- Tests lower resolution (384x512)
- Tests fewer inference steps (2 vs 4)
- Combines all optimizations
- Calculates final speedup

### Experiment 12: Memory Profiling
- Tracks GPU memory at each stage
- Reports peak memory usage
- Recommends minimum GPU requirements

### Final Summary
- Comparison table of all approaches
- Key findings and recommendations
- Next steps for implementation

## Expected Results

Based on typical GPU performance:

| Approach | Per Image | 100 Garments | Quality | VRAM | Recommended |
|----------|-----------|--------------|---------|------|-------------|
| SD 1.5 (30 steps) | ~8s | 13+ min | Excellent | 6 GB | ❌ |
| SD 1.5 (10 steps) | ~3s | 5+ min | Good | 6 GB | ❌ |
| LCM (4 steps) | ~0.4s | ~40s | Good | 8 GB | ✅ |
| LCM (2 steps) | ~0.25s | ~25s | Acceptable | 8 GB | ✅ |
| LCM optimized | ~0.15s | ~15s | Acceptable | 8 GB | ⭐ |

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch size in Experiment 8
- Use lower resolution (384x512)
- Close other GPU applications
- Use GPU with more VRAM

### Models Not Loading
- Check internet connection (models download from HuggingFace)
- Verify HuggingFace access (some models need login)
- Check disk space (models are several GB each)

### Slow Performance
- Ensure CUDA is properly installed
- Check GPU is being used: `torch.cuda.is_available()`
- Install xformers for better performance
- Close background applications

### Import Errors
- Reinstall packages: `pip install --upgrade [package-name]`
- Check Python version (3.10+ required)
- Create fresh virtual environment if needed

## Next Steps After Running

1. **Review Results**: Check actual timings on your GPU
2. **Assess Quality**: Visual inspection of generated images
3. **Choose Approach**: 
   - If <30s is OK: Use LCM with 4 steps
   - If need <10s: Implement template pre-rendering
4. **Optimize Further**: Based on Experiment 11 results
5. **Test with Real Data**: Use actual person/garment images

## Output Files

All experiment outputs are saved to:
```
outputs/
├── experiment_1/
├── experiment_2/
│   ├── person.png
│   └── garment.png
├── experiment_3/
│   ├── mask.png
│   └── segmented.png
├── experiment_4/
│   ├── mask.png
│   └── clean_garment.png
├── experiment_5/
│   └── pose.png
├── experiment_6/
│   ├── sd15_10steps.png
│   ├── sd15_20steps.png
│   └── sd15_30steps.png
├── experiment_7/
│   ├── lcm_1steps.png
│   ├── lcm_2steps.png
│   ├── lcm_4steps.png
│   └── lcm_8steps.png
└── experiment_8-12/
```

## Contact & Support

For issues or questions about these experiments:
- Review the notebook cell outputs for specific error messages
- Check GPU compatibility and driver versions
- Verify all dependencies are installed correctly

## License

This experimental notebook is part of the Tivora VTO project.

