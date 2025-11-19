import sys
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
catvton_path = os.path.join(os.path.dirname(__file__), '..', 'CatVTON')
sys.path.insert(0, catvton_path)

from efficient_batch_tryon import EfficientBatchVTON


class SAMBasedVTON:
    """
    Virtual Try-On using SAM 2.1 for accurate body segmentation
    """
    
    def __init__(self):
        """Initialize with SAM 2.1 and VTON pipeline"""
        print("🚀 Initializing SAM-Based VTON Pipeline...")
        
        # Initialize VTON pipeline
        self.vton = EfficientBatchVTON(
            base_model_path="booksforcharlie/stable-diffusion-inpainting",
            resume_path="zhengchong/CatVTON",
            dataset_name="vitonhd",
            mixed_precision="fp16",
            device="cuda",
            compile_model=False,
        )
        
        # Load SAM 2.1
        self.sam_predictor = self._load_sam()
    
    def _load_sam(self):
        """Load SAM 2.1 for segmentation"""
        try:
            print("Loading SAM 2.1 for accurate segmentation...")
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-small",
                device="cuda"
            )
            
            print("✅ SAM 2.1 loaded successfully")
            return predictor
        except Exception as e:
            print(f"❌ Error loading SAM 2.1: {e}")
            print("   Please ensure SAM 2.1 is installed:")
            print("   pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            return None
    
    def segment_person(self, image_path):
        """
        Segment person from image using SAM 2.1
        
        Args:
            image_path: Path to person image
        
        Returns:
            person_mask: Binary mask of the person (PIL Image, L mode)
        """
        if self.sam_predictor is None:
            raise RuntimeError("SAM 2.1 not loaded. Cannot segment person.")
        
        print("🎭 Segmenting person using SAM 2.1...")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Set image for SAM
        self.sam_predictor.set_image(image_array)
        
        # Use center point as prompt (assuming person is centered)
        h, w = image_array.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        center_label = np.array([1])
        
        # Predict mask
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=False
        )
        
        # Get the mask
        person_mask = masks[0].astype(bool)
        
        # Convert to PIL Image
        mask_img = Image.fromarray((person_mask * 255).astype(np.uint8), mode='L')
        
        print("   ✅ Person segmented successfully")
        return mask_img, image
    
    def create_clothing_mask(self, person_mask_pil, image_size):
        """
        Create agnostic mask for clothing area based on person segmentation
        
        Strategy:
        1. Get person segmentation from SAM
        2. Identify torso region (upper body)
        3. Create mask for clothing area only
        4. Preserve face, hands, and lower body
        
        Args:
            person_mask_pil: Person segmentation mask (PIL Image)
            image_size: (width, height) tuple
        
        Returns:
            clothing_mask: Mask for clothing area (PIL Image, L mode)
        """
        print("👕 Creating clothing-specific mask...")
        
        w, h = image_size
        person_mask = np.array(person_mask_pil)
        
        # Create clothing mask (initially same as person mask)
        clothing_mask = person_mask.copy()
        
        # Define regions to PRESERVE (not replace):
        # 1. Head/Face region (top 25% of image)
        head_cutoff = int(h * 0.25)
        clothing_mask[:head_cutoff, :] = 0
        
        # 2. Lower body region (bottom 40% of image) - preserve legs
        lower_cutoff = int(h * 0.60)
        clothing_mask[lower_cutoff:, :] = 0
        
        # 3. Hands/Arms - remove outer edges to preserve hands
        # Left edge
        left_preserve = int(w * 0.15)
        clothing_mask[:, :left_preserve] = 0
        
        # Right edge  
        right_preserve = int(w * 0.85)
        clothing_mask[:, right_preserve:] = 0
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        
        # Close small holes
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Open to remove small regions
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Smooth the edges with Gaussian blur
        clothing_mask = cv2.GaussianBlur(clothing_mask, (15, 15), 0)
        
        # Threshold back to binary
        _, clothing_mask = cv2.threshold(clothing_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Convert back to PIL
        mask_pil = Image.fromarray(clothing_mask.astype(np.uint8), mode='L')
        
        print("   ✅ Clothing mask created")
        return mask_pil
    
    def preprocess_images(self, person_path, garment_path, target_size=(768, 1024)):
        """
        Preprocess person and garment images
        
        Args:
            person_path: Path to person image
            garment_path: Path to garment image
            target_size: Target size (width, height)
        
        Returns:
            person_img, garment_img, clothing_mask
        """
        print("\n📸 Preprocessing images...")
        
        # Segment person first
        person_mask, person_img = self.segment_person(person_path)
        
        # Resize person image and mask
        person_img = person_img.resize(target_size, Image.Resampling.LANCZOS)
        person_mask = person_mask.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create clothing-specific mask
        clothing_mask = self.create_clothing_mask(person_mask, target_size)
        
        # Process garment
        garment_img = Image.open(garment_path).convert('RGB')
        
        # Resize garment preserving aspect ratio
        garment_img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Paste on white background
        garment_bg = Image.new('RGB', target_size, (255, 255, 255))
        offset = ((target_size[0] - garment_img.size[0]) // 2,
                  (target_size[1] - garment_img.size[1]) // 2)
        garment_bg.paste(garment_img, offset)
        
        print("   ✅ Images preprocessed")
        
        return person_img, garment_bg, clothing_mask, person_mask
    
    def visualize_mask_overlay(self, image, mask, output_path):
        """
        Visualize mask overlaid on image for debugging
        
        Args:
            image: PIL Image
            mask: PIL Image (L mode)
            output_path: Where to save
        """
        # Convert to numpy
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Create colored overlay (red for mask region)
        overlay = img_array.copy()
        overlay[mask_array > 128] = [255, 0, 0]
        
        # Blend
        result = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
        
        # Save
        Image.fromarray(result).save(output_path)
        print(f"   Saved mask visualization: {output_path}")
    
    def sam_based_tryon(
        self,
        person_image_path,
        garment_image_path,
        output_path="output_sam_based.png",
        resolution=(768, 1024),
        num_inference_steps=75,
        guidance_scale=2.5,
        seed=42,
        save_debug=True,
    ):
        """
        Perform virtual try-on using SAM-based segmentation
        
        Args:
            person_image_path: Path to person image
            garment_image_path: Path to garment image
            output_path: Where to save result
            resolution: (width, height)
            num_inference_steps: Quality vs speed (50-100)
            guidance_scale: How closely to follow garment (2.0-4.0)
            seed: Random seed
            save_debug: Whether to save debug visualizations
        
        Returns:
            PIL Image result
        """
        print("\n" + "="*70)
        print("🎨 SAM-BASED VIRTUAL TRY-ON")
        print("="*70)
        
        # Preprocess images with SAM segmentation
        person_img, garment_img, clothing_mask, person_mask = self.preprocess_images(
            person_image_path,
            garment_image_path,
            resolution
        )
        
        # Save debug images if requested
        if save_debug:
            print("\n💾 Saving debug images...")
            person_img.save('sam_debug_person.png')
            garment_img.save('sam_debug_garment.png')
            clothing_mask.save('sam_debug_clothing_mask.png')
            person_mask.save('sam_debug_person_mask.png')
            
            # Create mask overlay visualization
            self.visualize_mask_overlay(person_img, clothing_mask, 'sam_debug_mask_overlay.png')
            
            print("   ✅ Debug images saved:")
            print("      • sam_debug_person.png - Preprocessed person")
            print("      • sam_debug_garment.png - Preprocessed garment")
            print("      • sam_debug_person_mask.png - Full person segmentation")
            print("      • sam_debug_clothing_mask.png - Clothing area only")
            print("      • sam_debug_mask_overlay.png - Mask overlaid on person")
        
        # Run try-on
        print(f"\n🔧 Try-On Settings:")
        print(f"   • Resolution: {resolution[0]}x{resolution[1]}")
        print(f"   • Inference steps: {num_inference_steps}")
        print(f"   • Guidance scale: {guidance_scale}")
        print(f"   • Seed: {seed}")
        
        print("\n⏳ Running virtual try-on...")
        print("   (This will take 40-90 seconds depending on settings)")
        
        result = self.vton.single_try_on(
            person_image=person_img,
            garment_image=garment_img,
            mask=clothing_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=resolution[0],
            height=resolution[1],
            seed=seed,
        )
        
        # Save result
        result.save(output_path)
        
        print(f"\n✅ Result saved to: {output_path}")
        print("="*70)
        
        return result
    
    def batch_quality_test(
        self,
        person_image_path,
        garment_image_path,
        output_dir="sam_quality_tests"
    ):
        """
        Test multiple quality settings with SAM-based masking
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("🧪 TESTING MULTIPLE SETTINGS WITH SAM MASKING")
        print("="*70)
        
        configs = [
            {"steps": 50, "guidance": 2.0, "name": "fast"},
            {"steps": 75, "guidance": 2.5, "name": "balanced"},
            {"steps": 75, "guidance": 3.0, "name": "high_guidance"},
            {"steps": 100, "guidance": 2.5, "name": "ultra_quality"},
        ]
        
        results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Testing: {config['name']}")
            
            output_path = os.path.join(output_dir, f"sam_{config['name']}.png")
            
            result = self.sam_based_tryon(
                person_image_path=person_image_path,
                garment_image_path=garment_image_path,
                output_path=output_path,
                resolution=(768, 1024),
                num_inference_steps=config['steps'],
                guidance_scale=config['guidance'],
                seed=42,
                save_debug=(i == 1),  # Only save debug for first run
            )
            
            results.append((config['name'], result))
        
        print("\n" + "="*70)
        print(f"✅ All tests complete! Results in '{output_dir}' folder")
        print("="*70)
        
        return results


if __name__ == "__main__":
    # Initialize SAM-based pipeline
    sam_vton = SAMBasedVTON()
    
    # Single try-on with SAM-based masking
    result = sam_vton.sam_based_tryon(
        person_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
        garment_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\tshirt02.png",
        output_path="output_sam_based_latest.png",
        resolution=(768, 1024),
        num_inference_steps=75,
        guidance_scale=2.5,
        seed=42,
        save_debug=True,
    )
    
    print("\n🎉 Done! Check these files:")
    print("   • output_sam_based.png - Final result with SAM masking")
    print("   • sam_debug_mask_overlay.png - See the mask on the person")
    print("\n💡 The mask should now follow your body contours accurately!")
    print("\n💡 To test multiple settings, uncomment:")
    print("   # sam_vton.batch_quality_test(...)")
    
    # Uncomment to test multiple settings:
    # sam_vton.batch_quality_test(
    #     person_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
    #     garment_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\tshirt02.png",
    #     output_dir="sam_quality_tests"
    # )

