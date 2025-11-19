import os
import sys
import torch
import torch._dynamo
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor

# Add CatVTON to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
catvton_path = os.path.join(script_dir, '..', 'CatVTON')
if os.path.exists(catvton_path):
    sys.path.insert(0, catvton_path)

from model.pipeline import CatVTONPipeline


class EfficientBatchVTON:
    """
    Virtual try-on pipeline for processing multiple garments
    
    Note: CatVTON processes garments sequentially (one at a time).
    Typical performance: ~2-3 seconds per garment at 512x384 resolution.
    """
    
    def __init__(
        self,
        base_model_path: str = "booksforcharlie/stable-diffusion-inpainting",
        resume_path: str = "zhengchong/CatVTON",
        dataset_name: str = "vitonhd",
        mixed_precision: str = "fp16",  # Use fp16 for speed
        device: str = "cuda",
        compile_model: bool = False,  # Disabled by default (Triton issues on Windows)
    ):
        """
        Initialize the efficient batch processing pipeline
        
        Args:
            base_model_path: Base stable diffusion model path
            resume_path: CatVTON checkpoint path
            dataset_name: Dataset type (vitonhd/dresscode)
            mixed_precision: Precision mode (fp16/bf16/no)
            device: Device to run on
            compile_model: Whether to compile model (Linux only - requires Triton)
        
        Note:
            On Windows, keep compile_model=False as Triton is not well supported.
        """
        print("🚀 Initializing Efficient Batch VTON Pipeline...")
        
        # Set optimal settings for speed
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Suppress torch compilation errors (especially on Windows)
        if not compile_model:
            torch._dynamo.config.suppress_errors = True
        
        # Initialize pipeline with optimizations
        self.pipeline = CatVTONPipeline(
            attn_ckpt_version=dataset_name,
            attn_ckpt=resume_path,
            base_ckpt=base_model_path,
            weight_dtype=self._get_dtype(mixed_precision),
            device=device,
            skip_safety_check=True,  # Skip for speed
            compile=compile_model,
            use_tf32=True,
        )
        
        self.device = device
        self.dtype = self._get_dtype(mixed_precision)
        
        print("✅ Pipeline initialized successfully!")
    
    def _get_dtype(self, precision: str) -> torch.dtype:
        """Get torch dtype from precision string"""
        dtype_map = {
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(precision, torch.float16)
    
    def preprocess_images(
        self,
        person_image_path: str,
        garment_paths: List[str],
        mask_path: str = None,
        width: int = 384,
        height: int = 512,
    ) -> tuple:
        """
        Preprocess person and garment images efficiently
        
        Args:
            person_image_path: Path to person image
            garment_paths: List of garment image paths
            mask_path: Path to agnostic mask (auto-generated if None)
            width: Target width
            height: Target height
        
        Returns:
            Preprocessed tensors
        """
        # Load person image once
        person_img = Image.open(person_image_path).convert('RGB')
        person_img = person_img.resize((width, height), Image.LANCZOS)
        
        # Load mask (generate simple mask if not provided)
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            # Create a simple torso mask (you may want to use a proper mask generator)
            mask = self._generate_simple_mask(width, height)
        
        mask = mask.resize((width, height), Image.LANCZOS)
        
        # Load all garments in parallel
        garment_images = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            garment_images = list(executor.map(
                lambda p: Image.open(p).convert('RGB').resize((width, height), Image.LANCZOS),
                garment_paths
            ))
        
        return person_img, garment_images, mask
    
    def _generate_simple_mask(self, width: int, height: int) -> Image.Image:
        """Generate a simple mask for the torso region"""
        mask = Image.new('L', (width, height), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        # Simple torso mask (adjust as needed)
        draw.rectangle(
            [(width * 0.2, height * 0.15), (width * 0.8, height * 0.75)],
            fill=255
        )
        return mask
    
    def batch_try_on(
        self,
        person_image: Union[str, Image.Image],
        garment_images: List[Union[str, Image.Image]],
        mask: Union[str, Image.Image] = None,
        batch_size: int = 1,  # Note: CatVTON processes sequentially (kept for API compatibility)
        num_inference_steps: int = 25,  # Reduced for speed
        guidance_scale: float = 2.5,
        width: int = 384,
        height: int = 512,
        seed: int = 42,
        output_dir: str = "output_batch",
    ) -> List[Image.Image]:
        """
        Perform virtual try-on with multiple garments
        
        Note: CatVTON processes images sequentially, not in true batches.
        
        Args:
            person_image: Person image or path
            garment_images: List of garment images or paths
            mask: Mask image or path
            batch_size: (Deprecated - kept for API compatibility, not used)
            num_inference_steps: Number of denoising steps (lower = faster)
            guidance_scale: CFG scale
            width: Output width
            height: Output height
            seed: Random seed
            output_dir: Directory to save results
        
        Returns:
            List of result images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess inputs
        if isinstance(person_image, str):
            person_image = Image.open(person_image).convert('RGB')
        if isinstance(mask, str) and mask:
            mask = Image.open(mask).convert('L')
        elif mask is None:
            mask = self._generate_simple_mask(width, height)
        
        person_image = person_image.resize((width, height), Image.LANCZOS)
        mask = mask.resize((width, height), Image.LANCZOS)
        
        # Load garment images
        garments = []
        for g in garment_images:
            if isinstance(g, str):
                g = Image.open(g).convert('RGB')
            garments.append(g.resize((width, height), Image.LANCZOS))
        
        # Setup generator
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Process garments one by one
        all_results = []
        
        print(f"🎨 Processing {len(garments)} garments...")
        start_time = time.time()
        
        # Process each garment individually (CatVTON doesn't support true batching)
        for idx in tqdm(range(len(garments)), desc="Processing Try-Ons"):
            garment = garments[idx]
            
            # Run inference for single garment
            with torch.amp.autocast('cuda', enabled=True, dtype=self.dtype):
                result = self.pipeline(
                    image=person_image,
                    condition_image=garment,
                    mask=mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                )
            
            # Extract image from result (pipeline returns a list or the image directly)
            if isinstance(result, list):
                result = result[0]
            
            # Save result
            output_path = os.path.join(output_dir, f"result_{idx:04d}.png")
            result.save(output_path)
            all_results.append(result)
        
        elapsed_time = time.time() - start_time
        throughput = len(garments) / elapsed_time * 60  # garments per minute
        
        print(f"✅ Processed {len(garments)} garments in {elapsed_time:.2f}s")
        print(f"📊 Throughput: {throughput:.1f} garments/minute")
        
        return all_results
    
    def single_try_on(
        self,
        person_image: Union[str, Image.Image],
        garment_image: Union[str, Image.Image],
        mask: Union[str, Image.Image] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        width: int = 384,
        height: int = 512,
        seed: int = 42,
    ) -> Image.Image:
        """
        Perform single virtual try-on
        """
        results = self.batch_try_on(
            person_image=person_image,
            garment_images=[garment_image],
            mask=mask,
            batch_size=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
        )
        return results[0]
    
    def benchmark(
        self,
        person_image: str,
        garment_dir: str,
        mask: str = None,
        target_garments: int = 60,
        batch_size: int = 1,  # Not used, kept for API compatibility
        num_inference_steps: int = 25,
    ):
        """
        Benchmark the pipeline performance
        
        Args:
            person_image: Path to person image
            garment_dir: Directory containing garment images
            mask: Optional mask path
            target_garments: Number of garments to test with
            batch_size: (Not used - kept for compatibility)
            num_inference_steps: Number of inference steps
        """
        # Get garment files
        garment_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            garment_files.extend(Path(garment_dir).glob(ext))
        
        # Repeat if needed to reach target count
        while len(garment_files) < target_garments:
            garment_files.extend(garment_files)
        garment_files = garment_files[:target_garments]
        
        print(f"🔥 Benchmarking with {len(garment_files)} garments...")
        
        # Run benchmark
        results = self.batch_try_on(
            person_image=person_image,
            garment_images=[str(f) for f in garment_files],
            mask=mask,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            output_dir="benchmark_output"
        )
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Efficient Batch Virtual Try-On")
    
    # Model arguments
    parser.add_argument("--base_model_path", type=str, 
                       default="booksforcharlie/stable-diffusion-inpainting")
    parser.add_argument("--resume_path", type=str, 
                       default="zhengchong/CatVTON")
    parser.add_argument("--dataset_name", type=str, default="vitonhd",
                       choices=["vitonhd", "dresscode"])
    
    # Input arguments
    parser.add_argument("--person_image", type=str, required=True,
                       help="Path to person image")
    parser.add_argument("--garment_dir", type=str, required=True,
                       help="Directory containing garment images")
    parser.add_argument("--mask", type=str, default=None,
                       help="Path to agnostic mask (optional)")
    
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                       help="Number of inference steps (lower = faster)")
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="output_batch")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"])
    
    # Mode
    parser.add_argument("--benchmark", action="store_true",
                       help="Run in benchmark mode")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    vton = EfficientBatchVTON(
        base_model_path=args.base_model_path,
        resume_path=args.resume_path,
        dataset_name=args.dataset_name,
        mixed_precision=args.mixed_precision,
    )
    
    # Get garment images
    garment_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
        garment_files.extend(Path(args.garment_dir).glob(ext))
    garment_files = sorted(garment_files)
    
    print(f"📁 Found {len(garment_files)} garment images")
    
    if args.benchmark:
        # Run benchmark
        vton.benchmark(
            person_image=args.person_image,
            garment_dir=args.garment_dir,
            mask=args.mask,
            target_garments=60,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
        )
    else:
        # Run normal batch processing
        results = vton.batch_try_on(
            person_image=args.person_image,
            garment_images=[str(f) for f in garment_files],
            mask=args.mask,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        
        print(f"✨ Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()