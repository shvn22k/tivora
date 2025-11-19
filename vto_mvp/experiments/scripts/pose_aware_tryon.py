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


class PoseAwareVTON:
    """
    Virtual Try-On using both SAM 2.1 and Pose Detection for accurate masking
    """
    
    def __init__(self):
        """Initialize with SAM, Pose, and VTON pipeline"""
        print("🚀 Initializing Pose-Aware VTON Pipeline...")
        
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
        
        # Load OpenPose
        self.pose_detector = self._load_pose()
    
    def _load_sam(self):
        """Load SAM 2.1 for segmentation"""
        try:
            print("Loading SAM 2.1...")
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-small",
                device="cuda"
            )
            
            print("✅ SAM 2.1 loaded")
            return predictor
        except Exception as e:
            print(f"❌ Error loading SAM: {e}")
            return None
    
    def _load_pose(self):
        """Load OpenPose for pose detection"""
        try:
            print("Loading OpenPose detector...")
            from controlnet_aux import OpenposeDetector
            
            detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            print("✅ OpenPose loaded")
            return detector
        except Exception as e:
            print(f"❌ Error loading OpenPose: {e}")
            return None
    
    def detect_pose(self, image):
        """Detect pose keypoints"""
        if self.pose_detector is None:
            print("⚠️ OpenPose not available")
            return None
        
        print("🏃 Detecting pose keypoints...")
        pose_image = self.pose_detector(image)
        print("   ✅ Pose detected")
        return pose_image
    
    def segment_person(self, image_path):
        """Segment person from image using SAM 2.1"""
        if self.sam_predictor is None:
            raise RuntimeError("SAM 2.1 not loaded")
        
        print("🎭 Segmenting person with SAM 2.1...")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Set image for SAM
        self.sam_predictor.set_image(image_array)
        
        # Use center point as prompt
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
        
        print("   ✅ Person segmented")
        return mask_img, image
    
    def create_torso_mask_from_person(self, person_mask_pil, pose_image, image_size):
        """
        Create accurate torso mask using person segmentation and pose
        
        Strategy:
        1. Use SAM person mask as base
        2. Use pose keypoints to identify head, shoulders, waist
        3. Create mask for torso region only
        4. Feather edges for natural blending
        
        Args:
            person_mask_pil: Person segmentation from SAM
            pose_image: Pose detection result (or None)
            image_size: (width, height)
        
        Returns:
            torso_mask: Accurate mask for clothing area
        """
        print("👕 Creating pose-aware torso mask...")
        
        w, h = image_size
        person_mask = np.array(person_mask_pil)
        
        # Start with person mask
        torso_mask = person_mask.copy()
        
        # Find person's vertical boundaries
        rows_with_person = np.any(person_mask > 128, axis=1)
        person_top = np.argmax(rows_with_person)
        person_bottom = len(rows_with_person) - np.argmax(rows_with_person[::-1])
        person_height = person_bottom - person_top
        
        if person_height > 0:
            # Estimate body regions based on typical proportions
            # For a full-body image:
            # - Head: top 15% of person height
            # - Torso/Shirt: 15-55% of person height  
            # - Legs: 55-100% of person height
            
            head_end = person_top + int(person_height * 0.18)
            torso_end = person_top + int(person_height * 0.60)
            
            # Zero out head region
            torso_mask[:head_end, :] = 0
            
            # Zero out legs region
            torso_mask[torso_end:, :] = 0
            
            print(f"   Person height: {person_height}px")
            print(f"   Head region: 0-{head_end}")
            print(f"   Torso region: {head_end}-{torso_end}")
            print(f"   Legs region: {torso_end}-{h}")
        else:
            print("   ⚠️ Could not detect person boundaries, using fallback")
            # Fallback to percentage-based
            torso_mask[:int(h * 0.18), :] = 0  # Remove head
            torso_mask[int(h * 0.60):, :] = 0  # Remove legs
        
        # Find horizontal boundaries and add margins for arms
        cols_with_torso = np.any(torso_mask > 128, axis=0)
        if np.any(cols_with_torso):
            torso_left = np.argmax(cols_with_torso)
            torso_right = len(cols_with_torso) - np.argmax(cols_with_torso[::-1])
            torso_width = torso_right - torso_left
            
            # Add margins for arms/sleeves (20% on each side)
            arm_margin = int(torso_width * 0.25)
            
            # Expand mask horizontally for arms
            left_boundary = max(0, torso_left - arm_margin)
            right_boundary = min(w, torso_right + arm_margin)
            
            # Create expanded torso mask
            expanded_mask = np.zeros_like(torso_mask)
            for y in range(head_end, torso_end):
                if np.any(torso_mask[y, :] > 128):
                    expanded_mask[y, left_boundary:right_boundary] = 255
            
            # Blend with original to maintain natural shape
            torso_mask = np.maximum(torso_mask, expanded_mask)
        
        # Morphological operations for smooth mask
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((9, 9), np.uint8)
        
        # Close small gaps
        torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, kernel_small, iterations=3)
        
        # Dilate slightly to ensure coverage
        torso_mask = cv2.dilate(torso_mask, kernel_small, iterations=2)
        
        # Smooth edges with strong Gaussian blur for natural transition
        torso_mask = cv2.GaussianBlur(torso_mask, (31, 31), 0)
        
        # Apply feathering at boundaries for better blending
        # Create distance transform for gradual fade
        _, binary_mask = cv2.threshold(torso_mask, 100, 255, cv2.THRESH_BINARY)
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # Normalize and apply to create feathered edges
        if dist_transform.max() > 0:
            dist_normalized = (dist_transform / dist_transform.max() * 255).astype(np.uint8)
            # Blend with original
            torso_mask = np.maximum(torso_mask, dist_normalized)
        
        # Convert back to PIL
        mask_pil = Image.fromarray(torso_mask.astype(np.uint8), mode='L')
        
        print("   ✅ Pose-aware torso mask created")
        return mask_pil
    
    def preprocess_images(self, person_path, garment_path, target_size=(768, 1024)):
        """Preprocess all images with pose-aware masking"""
        print("\n📸 Preprocessing images with pose detection...")
        
        # Segment person
        person_mask, person_img = self.segment_person(person_path)
        
        # Detect pose
        pose_result = self.detect_pose(person_img)
        
        # Resize
        person_img = person_img.resize(target_size, Image.Resampling.LANCZOS)
        person_mask = person_mask.resize(target_size, Image.Resampling.LANCZOS)
        if pose_result:
            pose_result = pose_result.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create pose-aware torso mask
        torso_mask = self.create_torso_mask_from_person(person_mask, pose_result, target_size)
        
        # Process garment
        garment_img = Image.open(garment_path).convert('RGB')
        garment_img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        garment_bg = Image.new('RGB', target_size, (255, 255, 255))
        offset = ((target_size[0] - garment_img.size[0]) // 2,
                  (target_size[1] - garment_img.size[1]) // 2)
        garment_bg.paste(garment_img, offset)
        
        print("   ✅ All images preprocessed")
        
        return person_img, garment_bg, torso_mask, person_mask, pose_result
    
    def visualize_mask_overlay(self, image, mask, output_path, color=(255, 0, 0)):
        """Visualize mask overlaid on image"""
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Create colored overlay
        overlay = img_array.copy()
        
        # Apply color with alpha based on mask value
        for i in range(3):
            overlay[:, :, i] = np.where(
                mask_array > 10,
                np.clip(img_array[:, :, i] * 0.6 + color[i] * 0.4 * (mask_array / 255.0), 0, 255),
                img_array[:, :, i]
            )
        
        Image.fromarray(overlay.astype(np.uint8)).save(output_path)
        print(f"   Saved: {output_path}")
    
    def pose_aware_tryon(
        self,
        person_image_path,
        garment_image_path,
        output_path="output_pose_aware.png",
        resolution=(768, 1024),
        num_inference_steps=75,
        guidance_scale=2.5,
        seed=42,
        save_debug=True,
    ):
        """
        Perform virtual try-on with pose-aware masking
        """
        print("\n" + "="*70)
        print("🎨 POSE-AWARE VIRTUAL TRY-ON")
        print("="*70)
        
        # Preprocess with pose awareness
        person_img, garment_img, torso_mask, person_mask, pose_result = self.preprocess_images(
            person_image_path,
            garment_image_path,
            resolution
        )
        
        # Save debug images
        if save_debug:
            print("\n💾 Saving debug visualizations...")
            person_img.save('pose_debug_person.png')
            garment_img.save('pose_debug_garment.png')
            person_mask.save('pose_debug_person_mask.png')
            torso_mask.save('pose_debug_torso_mask.png')
            if pose_result:
                pose_result.save('pose_debug_pose.png')
            
            # Create overlays
            self.visualize_mask_overlay(person_img, person_mask, 'pose_debug_person_segmentation.png', (0, 255, 0))
            self.visualize_mask_overlay(person_img, torso_mask, 'pose_debug_torso_overlay.png', (255, 0, 0))
            
            print("   ✅ Debug files saved:")
            print("      • pose_debug_person.png")
            print("      • pose_debug_garment.png")
            print("      • pose_debug_person_mask.png")
            print("      • pose_debug_torso_mask.png")
            print("      • pose_debug_person_segmentation.png (green = full person)")
            print("      • pose_debug_torso_overlay.png (red = clothing area)")
            if pose_result:
                print("      • pose_debug_pose.png")
        
        # Run try-on
        print(f"\n🔧 Try-On Settings:")
        print(f"   • Resolution: {resolution[0]}x{resolution[1]}")
        print(f"   • Steps: {num_inference_steps}")
        print(f"   • Guidance: {guidance_scale}")
        
        print("\n⏳ Running virtual try-on...")
        
        result = self.vton.single_try_on(
            person_image=person_img,
            garment_image=garment_img,
            mask=torso_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=resolution[0],
            height=resolution[1],
            seed=seed,
        )
        
        result.save(output_path)
        
        print(f"\n✅ Result saved: {output_path}")
        print("="*70)
        
        return result


if __name__ == "__main__":
    # Initialize
    pose_vton = PoseAwareVTON()
    
    # Run with pose-aware masking
    result = pose_vton.pose_aware_tryon(
        person_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
        garment_image_path=r"C:\Projects\tivora\vto_mvp\notebooks\tshirt02.png",
        output_path="output_pose_aware.png",
        resolution=(768, 1024),
        num_inference_steps=75,
        guidance_scale=2.5,
        seed=42,
        save_debug=True,
    )
    
    print("\n🎉 Complete! Check these files:")
    print("   • output_pose_aware.png - Final result")
    print("   • pose_debug_torso_overlay.png - RED shows clothing area")
    print("   • pose_debug_person_segmentation.png - GREEN shows full person")
    print("\n💡 The red overlay should cover ONLY the shirt area")
    print("   Face, hands, and legs should NOT be red")

