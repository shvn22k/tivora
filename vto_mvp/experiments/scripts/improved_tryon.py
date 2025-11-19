import sys
import os
import torch
import numpy as np
from PIL import Image
import cv2

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
catvton_path = os.path.join(os.path.dirname(__file__), '..', 'CatVTON')
sys.path.insert(0, catvton_path)

from efficient_batch_tryon import EfficientBatchVTON


class ImprovedVTON:
    """
    Improved Virtual Try-On with better preprocessing and quality
    """
    
    def __init__(self):
        """Initialize with high-quality settings"""
        print("🚀 Initializing Improved VTON Pipeline...")
        
        self.vton = EfficientBatchVTON(
            base_model_path="booksforcharlie/stable-diffusion-inpainting",
            resume_path="zhengchong/CatVTON",
            dataset_name="vitonhd",
            mixed_precision="fp16",
            device="cuda",
            compile_model=False,
        )
        
        # Try to load segmentation model for better masks
        self.seg_model = self._load_segmentation_model()
    
    def _load_segmentation_model(self):
        """Load segmentation model for generating proper agnostic masks"""
        try:
            from transformers import AutoModelForImageSegmentation
            from torchvision import transforms
            
            print("Loading segmentation model for mask generation...")
            model = AutoModelForImageSegmentation.from_pretrained(
                'briaai/RMBG-1.4',
                trust_remote_code=True
            )
            model.to('cuda')
            model.eval()
            
            self.seg_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            print("✅ Segmentation model loaded")
            return model
        except Exception as e:
            print(f"⚠️ Could not load segmentation model: {e}")
            print("   Will use simple mask generation instead")
            return None
    
    def preprocess_person_image(self, image_path, target_size=(768, 1024)):
        """
        Preprocess person image with proper cropping and centering
        
        Args:
            image_path: Path to person image
            target_size: (width, height)
        
        Returns:
            Preprocessed PIL Image
        """
        print("📸 Preprocessing person image...")
        img = Image.open(image_path).convert('RGB')
        
        # Get current size
        w, h = img.size
        target_w, target_h = target_size
        
        # Calculate aspect ratios
        img_aspect = w / h
        target_aspect = target_w / target_h
        
        # Crop to target aspect ratio (center crop)
        if img_aspect > target_aspect:
            # Image is wider, crop width
            new_w = int(h * target_aspect)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        else:
            # Image is taller, crop height
            new_h = int(w / target_aspect)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
        
        # Resize to target size with high-quality resampling
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        print(f"   ✅ Resized to {target_size[0]}x{target_size[1]}")
        return img
    
    def preprocess_garment_image(self, image_path, target_size=(768, 1024)):
        """
        Preprocess garment image - remove background if needed
        
        Args:
            image_path: Path to garment image
            target_size: (width, height)
        
        Returns:
            Preprocessed PIL Image
        """
        print("👕 Preprocessing garment image...")
        img = Image.open(image_path).convert('RGB')
        
        # Remove background if segmentation model is available
        if self.seg_model is not None:
            try:
                print("   Removing background...")
                img = self._remove_background(img)
            except Exception as e:
                print(f"   ⚠️ Background removal failed: {e}")
        
        # Resize with aspect ratio preservation
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Paste on white background to ensure target size
        bg = Image.new('RGB', target_size, (255, 255, 255))
        offset = ((target_size[0] - img.size[0]) // 2, 
                  (target_size[1] - img.size[1]) // 2)
        bg.paste(img, offset)
        
        print(f"   ✅ Resized to {target_size[0]}x{target_size[1]}")
        return bg
    
    def _remove_background(self, image):
        """Remove background from image using segmentation model"""
        # Preprocess
        input_images = self.seg_transform(image).unsqueeze(0).to('cuda')
        
        # Predict
        with torch.no_grad():
            preds = self.seg_model(input_images)
            
            # Extract tensor from result
            pred = preds
            while isinstance(pred, (list, tuple)):
                pred = pred[-1]
            
            # Apply sigmoid if needed
            if pred.min() < 0 or pred.max() > 1:
                pred = pred.sigmoid()
            
            pred = pred.cpu()
        
        # Post-process
        pred = pred[0].squeeze()
        from torchvision import transforms
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        
        # Create image with white background
        result = Image.new('RGB', image.size, (255, 255, 255))
        result.paste(image, (0, 0), mask=mask)
        
        return result
    
    def generate_agnostic_mask(self, person_image, size=(768, 1024)):
        """
        Generate agnostic mask for the person (torso region)
        
        This is a simplified version. For production, use proper pose-based masking.
        
        Args:
            person_image: PIL Image of person
            size: (width, height)
        
        Returns:
            PIL Image mask (L mode)
        """
        print("🎭 Generating agnostic mask...")
        
        w, h = size
        mask = Image.new('L', size, 0)
        
        # Use OpenCV for better mask generation
        mask_array = np.zeros((h, w), dtype=np.uint8)
        
        # Define torso region for full body image
        # Upper body region where garment should be placed
        torso_top = int(h * 0.15)      # Start below head
        torso_bottom = int(h * 0.65)   # End at waist/hip
        torso_left = int(w * 0.25)     # Left side
        torso_right = int(w * 0.75)    # Right side
        
        # Create elliptical torso mask for more natural look
        center_x = w // 2
        center_y = (torso_top + torso_bottom) // 2
        axis_x = (torso_right - torso_left) // 2
        axis_y = (torso_bottom - torso_top) // 2
        
        cv2.ellipse(
            mask_array,
            (center_x, center_y),
            (axis_x, axis_y),
            0, 0, 360,
            255,
            -1
        )
        
        # Also add arms region (simplified)
        # Left arm
        cv2.ellipse(
            mask_array,
            (int(w * 0.20), int(h * 0.35)),
            (int(w * 0.08), int(h * 0.15)),
            -30, 0, 360,
            255,
            -1
        )
        
        # Right arm
        cv2.ellipse(
            mask_array,
            (int(w * 0.80), int(h * 0.35)),
            (int(w * 0.08), int(h * 0.15)),
            30, 0, 360,
            255,
            -1
        )
        
        # Apply Gaussian blur for smooth edges
        mask_array = cv2.GaussianBlur(mask_array, (21, 21), 0)
        
        mask = Image.fromarray(mask_array, mode='L')
        print("   ✅ Agnostic mask generated")
        
        return mask
    
    def high_quality_tryon(
        self,
        person_image_path,
        garment_image_path,
        output_path="output_hq.png",
        resolution=(768, 1024),
        num_inference_steps=50,
        guidance_scale=2.5,
        seed=42,
    ):
        """
        Perform high-quality virtual try-on
        
        Args:
            person_image_path: Path to person image
            garment_image_path: Path to garment image
            output_path: Where to save result
            resolution: (width, height) - use 768x1024 for full body
            num_inference_steps: More steps = better quality (50-100)
            guidance_scale: 2.0-4.0 range
            seed: Random seed
        
        Returns:
            PIL Image result
        """
        print("\n" + "="*60)
        print("🎨 HIGH QUALITY VIRTUAL TRY-ON")
        print("="*60)
        
        # Step 1: Preprocess images
        person_image = self.preprocess_person_image(person_image_path, resolution)
        garment_image = self.preprocess_garment_image(garment_image_path, resolution)
        
        # Step 2: Generate agnostic mask
        mask = self.generate_agnostic_mask(person_image, resolution)
        
        # Save preprocessed for inspection
        person_image.save('debug_person_preprocessed.png')
        garment_image.save('debug_garment_preprocessed.png')
        mask.save('debug_mask.png')
        print("\n📁 Saved debug images: debug_person_preprocessed.png, debug_garment_preprocessed.png, debug_mask.png")
        
        # Step 3: Run try-on with high quality settings
        print(f"\n🔧 Settings:")
        print(f"   • Resolution: {resolution[0]}x{resolution[1]}")
        print(f"   • Inference steps: {num_inference_steps}")
        print(f"   • Guidance scale: {guidance_scale}")
        print(f"   • Seed: {seed}")
        
        print("\n⏳ Running virtual try-on (this may take 30-60 seconds)...")
        
        result = self.vton.single_try_on(
            person_image=person_image,
            garment_image=garment_image,
            mask=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=resolution[0],
            height=resolution[1],
            seed=seed,
        )
        
        # Save result
        result.save(output_path)
        
        print(f"\n✅ Result saved to: {output_path}")
        print("="*60)
        
        return result
    
    def batch_test_settings(
        self,
        person_image_path,
        garment_image_path,
        output_dir="quality_tests"
    ):
        """
        Test different settings to find optimal quality
        
        This will generate multiple versions with different parameters
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("🧪 TESTING DIFFERENT QUALITY SETTINGS")
        print("="*60)
        
        # Test configurations
        configs = [
            {"steps": 50, "guidance": 2.0, "res": (768, 1024), "name": "standard"},
            {"steps": 75, "guidance": 2.5, "res": (768, 1024), "name": "high_steps"},
            {"steps": 50, "guidance": 3.0, "res": (768, 1024), "name": "high_guidance"},
            {"steps": 100, "guidance": 2.5, "res": (768, 1024), "name": "ultra_quality"},
        ]
        
        results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Testing: {config['name']}")
            print(f"   Steps: {config['steps']}, Guidance: {config['guidance']}, Res: {config['res']}")
            
            output_path = os.path.join(output_dir, f"test_{config['name']}.png")
            
            result = self.high_quality_tryon(
                person_image_path=person_image_path,
                garment_image_path=garment_image_path,
                output_path=output_path,
                resolution=config['res'],
                num_inference_steps=config['steps'],
                guidance_scale=config['guidance'],
                seed=42,
            )
            
            results.append((config['name'], result))
        
        print("\n" + "="*60)
        print(f"✅ All tests complete! Check the '{output_dir}' folder")
        print("="*60)
        
        return results


if __name__ == "__main__":
    # Initialize improved pipeline
    improved_vton = ImprovedVTON()
    
    # Single high-quality try-on
    result = improved_vton.high_quality_tryon(
        person_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
        garment_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\tshirt02.png",
        output_path="output_improved.png",
        resolution=(768, 1024),  # Higher resolution
        num_inference_steps=75,   # More steps for quality
        guidance_scale=2.5,
        seed=42,
    )
    
    print("\n🎉 Done! Compare:")
    print("   • output_single.png (old, low quality)")
    print("   • output_improved.png (new, improved)")
    print("\n💡 To test multiple settings, uncomment the batch test below:")
    print("   # improved_vton.batch_test_settings(...)")
    
    # Uncomment to test multiple settings:
    # improved_vton.batch_test_settings(
    #     person_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
    #     garment_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\tshirt02.png",
    #     output_dir="quality_tests"
    # )

