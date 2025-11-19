import sys
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
catvton_path = os.path.join(os.path.dirname(__file__), '..', 'CatVTON')
sys.path.insert(0, catvton_path)

from efficient_batch_tryon import EfficientBatchVTON


class ImprovedSAMVTON:
    """
    Improved Virtual Try-On with better clothing segmentation
    """
    
    def __init__(self):
        """Initialize with SAM 2.1 and VTON pipeline"""
        print("🚀 Initializing Improved SAM-Based VTON Pipeline...")
        
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
            print("📦 Loading SAM 2.1...")
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-small",
                device="cuda"
            )
            
            print("✅ SAM 2.1 loaded successfully")
            return predictor
        except Exception as e:
            print(f"❌ Error loading SAM 2.1: {e}")
            return None
    
    def segment_clothing_region(self, image_path, strategy="multi_point"):
        """
        Segment ONLY the clothing region (not full person)
        
        Args:
            image_path: Path to person image
            strategy: "multi_point", "box", or "torso_points"
        
        Returns:
            clothing_mask: Binary mask of clothing area only
        """
        if self.sam_predictor is None:
            raise RuntimeError("SAM 2.1 not loaded")
        
        print(f"👕 Segmenting clothing using '{strategy}' strategy...")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        
        # Set image for SAM
        self.sam_predictor.set_image(image_array)
        
        if strategy == "multi_point":
            # Use multiple points targeting the TORSO area only
            points = np.array([
                [w // 2, h * 0.35],      # Upper chest
                [w // 2, h * 0.45],      # Mid chest
                [w * 0.4, h * 0.40],     # Left chest
                [w * 0.6, h * 0.40],     # Right chest
                [w // 2, h * 0.55],      # Lower chest/stomach
            ])
            labels = np.array([1, 1, 1, 1, 1])  # All positive points
            
            # Add negative points to exclude face and legs
            negative_points = np.array([
                [w // 2, h * 0.15],      # Face (exclude)
                [w // 2, h * 0.80],      # Legs (exclude)
                [w * 0.2, h * 0.50],     # Arms (exclude)
                [w * 0.8, h * 0.50],     # Arms (exclude)
            ])
            
            # Combine points
            all_points = np.vstack([points, negative_points])
            all_labels = np.concatenate([labels, np.array([0, 0, 0, 0])])
            
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=all_points,
                point_labels=all_labels,
                multimask_output=False
            )
            
        elif strategy == "box":
            # Use bounding box around torso area
            box = np.array([
                w * 0.2,   # left
                h * 0.25,  # top (below head)
                w * 0.8,   # right
                h * 0.65   # bottom (above legs)
            ])
            
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=False
            )
            
        else:  # torso_points
            # Strategic points on torso with negative prompts
            center_y = h * 0.45  # Chest level
            
            points = np.array([
                [w // 2, center_y],           # Center chest
                [w * 0.35, center_y],         # Left torso
                [w * 0.65, center_y],         # Right torso
            ])
            labels = np.array([1, 1, 1])
            
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )
        
        # Get the mask
        clothing_mask = masks[0].astype(np.uint8) * 255
        
        # Post-process to refine mask
        clothing_mask = self._refine_clothing_mask(clothing_mask, image_array.shape[:2])
        
        # Convert to PIL
        mask_img = Image.fromarray(clothing_mask, mode='L')
        
        print("   ✅ Clothing region segmented")
        return mask_img, image
    
    def _refine_clothing_mask(self, mask, image_shape):
        """
        Refine the clothing mask to exclude unwanted regions
        
        Args:
            mask: numpy array (H, W) with values 0-255
            image_shape: (height, width)
        
        Returns:
            refined_mask: numpy array (H, W)
        """
        h, w = image_shape
        
        # 1. Remove head region (top 20%)
        head_cutoff = int(h * 0.20)
        mask[:head_cutoff, :] = 0
        
        # 2. Remove legs region (bottom 35%)
        legs_cutoff = int(h * 0.65)
        mask[legs_cutoff:, :] = 0
        
        # 3. Create elliptical constraint for torso
        ellipse_mask = np.zeros_like(mask)
        center_x, center_y = w // 2, int(h * 0.42)
        axes = (int(w * 0.30), int(h * 0.22))  # Narrower ellipse for torso
        cv2.ellipse(ellipse_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
        
        # Combine with existing mask
        mask = cv2.bitwise_and(mask, ellipse_mask)
        
        # 4. Morphological operations to smooth
        kernel = np.ones((7, 7), np.uint8)
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Dilate slightly to ensure coverage
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 5. Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # 6. Threshold back
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        
        # 7. Apply gradient mask for smooth blending
        mask = self._apply_gradient_edges(mask)
        
        return mask
    
    def _apply_gradient_edges(self, mask):
        """
        Apply gradient at the edges of mask for smoother blending
        """
        # Find edges
        edges = cv2.Canny(mask, 50, 150)
        
        # Create distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Normalize
        dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply soft gradient near edges
        gradient_width = 20  # pixels
        gradient_mask = np.zeros_like(mask)
        
        for i in range(gradient_width):
            alpha = 1.0 - (i / gradient_width)
            dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=i)
            gradient_mask = np.maximum(gradient_mask, (dilated * alpha * 255).astype(np.uint8))
        
        # Blend
        result = np.maximum(mask, gradient_mask)
        
        return result
    
    def create_hybrid_mask(self, image_path):
        """
        Create best quality mask using hybrid approach:
        1. Try all SAM strategies
        2. Combine with heuristic torso detection
        3. Return the best one
        """
        print("\n🔬 Creating hybrid mask with multiple strategies...")
        
        # Try different SAM strategies
        strategies = ["multi_point", "box", "torso_points"]
        masks = []
        scores = []
        
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        
        for strategy in strategies:
            try:
                mask, _ = self.segment_clothing_region(image_path, strategy=strategy)
                masks.append(np.array(mask))
                
                # Score mask quality (prefer masks in the torso region)
                torso_region = mask.crop((
                    int(w * 0.2), int(h * 0.25),
                    int(w * 0.8), int(h * 0.65)
                ))
                score = np.array(torso_region).mean()
                scores.append(score)
                
                print(f"   • {strategy}: score = {score:.1f}")
            except Exception as e:
                print(f"   ⚠️ {strategy} failed: {e}")
                continue
        
        # Select best mask
        if masks:
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            print(f"   ✅ Selected best strategy: {strategies[best_idx]}")
        else:
            # Fallback to heuristic
            print("   ⚠️ All SAM strategies failed, using heuristic mask")
            best_mask = self._create_heuristic_mask(image_array.shape[:2])
        
        return Image.fromarray(best_mask, mode='L'), image
    
    def _create_heuristic_mask(self, image_shape):
        """
        Fallback: Create mask using heuristic approach (no ML)
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create torso ellipse
        center_x, center_y = w // 2, int(h * 0.42)
        axes = (int(w * 0.28), int(h * 0.20))
        cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
        
        # Smooth
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def visualize_mask_overlay(self, image, mask, output_path):
        """Visualize mask overlaid on image"""
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Create red overlay
        overlay = img_array.copy()
        overlay[mask_array > 50] = [255, 0, 0]
        
        # Blend
        result = cv2.addWeighted(img_array, 0.6, overlay, 0.4, 0)
        
        # Add text
        cv2.putText(result, "Red = Will be replaced", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        Image.fromarray(result).save(output_path)
        print(f"   📊 Visualization saved: {output_path}")
    
    def preprocess_images(self, person_path, garment_path, target_size=(768, 1024), mask_strategy="hybrid"):
        """
        Preprocess images with improved masking
        
        Args:
            person_path: Path to person image
            garment_path: Path to garment image
            target_size: Target size (width, height)
            mask_strategy: "hybrid", "multi_point", "box", or "torso_points"
        
        Returns:
            person_img, garment_img, clothing_mask
        """
        print("\n📸 Preprocessing images with improved masking...")
        
        # Get clothing mask
        if mask_strategy == "hybrid":
            clothing_mask, person_img = self.create_hybrid_mask(person_path)
        else:
            clothing_mask, person_img = self.segment_clothing_region(person_path, strategy=mask_strategy)
        
        # Resize
        person_img = person_img.resize(target_size, Image.Resampling.LANCZOS)
        clothing_mask = clothing_mask.resize(target_size, Image.Resampling.LANCZOS)
        
        # Process garment
        garment_img = Image.open(garment_path).convert('RGB')
        garment_img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Center garment on white background
        garment_bg = Image.new('RGB', target_size, (255, 255, 255))
        offset = ((target_size[0] - garment_img.size[0]) // 2,
                  (target_size[1] - garment_img.size[1]) // 2)
        garment_bg.paste(garment_img, offset)
        
        print("   ✅ Images preprocessed with improved mask")
        
        return person_img, garment_bg, clothing_mask
    
    def improved_tryon(
        self,
        person_image_path,
        garment_image_path,
        output_path="output_improved.png",
        resolution=(768, 1024),
        num_inference_steps=75,
        guidance_scale=2.5,
        seed=42,
        mask_strategy="hybrid",
        save_debug=True,
    ):
        """
        Perform improved virtual try-on
        
        Args:
            person_image_path: Path to person image
            garment_image_path: Path to garment image
            output_path: Where to save result
            resolution: (width, height) - higher = better quality
            num_inference_steps: 50-100 (higher = better but slower)
            guidance_scale: 2.0-4.0 (how closely to follow garment)
            seed: Random seed for reproducibility
            mask_strategy: "hybrid", "multi_point", "box", or "torso_points"
            save_debug: Save debug visualizations
        
        Returns:
            PIL Image result
        """
        print("\n" + "="*70)
        print("🎨 IMPROVED SAM-BASED VIRTUAL TRY-ON")
        print("="*70)
        
        # Preprocess with improved masking
        person_img, garment_img, clothing_mask = self.preprocess_images(
            person_image_path,
            garment_image_path,
            resolution,
            mask_strategy=mask_strategy
        )
        
        # Save debug images
        if save_debug:
            print("\n💾 Saving debug images...")
            person_img.save('debug_person.png')
            garment_img.save('debug_garment.png')
            clothing_mask.save('debug_mask.png')
            self.visualize_mask_overlay(person_img, clothing_mask, 'debug_mask_overlay.png')
            
            print("   ✅ Debug images saved:")
            print("      • debug_mask_overlay.png - CHECK THIS to verify mask!")
            print("      • Red area = will be replaced with garment")
            print("      • Should ONLY cover torso, NOT face/arms/legs")
        
        # Settings info
        print(f"\n🔧 Try-On Settings:")
        print(f"   • Resolution: {resolution[0]}x{resolution[1]}")
        print(f"   • Inference steps: {num_inference_steps}")
        print(f"   • Guidance scale: {guidance_scale}")
        print(f"   • Mask strategy: {mask_strategy}")
        print(f"   • Seed: {seed}")
        
        print("\n⏳ Running virtual try-on (60-90 seconds)...")
        
        # Run try-on
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
        
        print(f"\n✅ Result saved: {output_path}")
        print("="*70)
        
        return result
    
    def compare_all_strategies(self, person_image_path, garment_image_path, output_dir="strategy_comparison"):
        """
        Compare all masking strategies side-by-side
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("🧪 COMPARING ALL MASKING STRATEGIES")
        print("="*70)
        
        strategies = ["hybrid", "multi_point", "box", "torso_points"]
        results = {}
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n[{i}/{len(strategies)}] Testing: {strategy}")
            
            try:
                result = self.improved_tryon(
                    person_image_path=person_image_path,
                    garment_image_path=garment_image_path,
                    output_path=os.path.join(output_dir, f"result_{strategy}.png"),
                    resolution=(768, 1024),
                    num_inference_steps=75,
                    guidance_scale=2.5,
                    seed=42,
                    mask_strategy=strategy,
                    save_debug=(i == 1),
                )
                results[strategy] = result
            except Exception as e:
                print(f"   ❌ {strategy} failed: {e}")
                continue
        
        print("\n" + "="*70)
        print(f"✅ Comparison complete! Check '{output_dir}' folder")
        print("   Compare results to see which strategy works best for your images")
        print("="*70)
        
        return results


if __name__ == "__main__":
    # Initialize
    vton = ImprovedSAMVTON()
    
    # Test single try-on with improved masking
    result = vton.improved_tryon(
        person_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
        garment_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\tshirt02.png",
        output_path="output_improved.png",
        resolution=(768, 1024),
        num_inference_steps=75,
        guidance_scale=2.5,
        seed=42,
        mask_strategy="hybrid",  # Try: "hybrid", "multi_point", "box", "torso_points"
        save_debug=True,
    )
    
    print("\n🎉 Done! Check these files:")
    print("   1. debug_mask_overlay.png - VERIFY the red region looks correct")
    print("   2. output_improved.png - Final result")
    print("\n💡 If mask is wrong:")
    print("   • Try different mask_strategy: 'hybrid', 'multi_point', 'box', 'torso_points'")
    print("   • Or run comparison to test all strategies:")
    print("     vton.compare_all_strategies(...)")
    
    # Uncomment to compare all strategies:
    # vton.compare_all_strategies(
    #     person_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
    #     garment_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\tshirt02.png",
    #     output_dir="strategy_comparison"
    # )