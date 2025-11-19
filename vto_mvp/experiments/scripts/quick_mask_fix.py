"""
Quick manual mask adjustments if SAM still doesn't work perfectly
"""
import numpy as np
from PIL import Image, ImageDraw
import cv2

def create_perfect_torso_mask(image_path, output_path="perfect_mask.png"):
    """
    Create a perfect torso mask without any ML
    Adjust the parameters based on your specific image
    """
    # Load image to get dimensions
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    # Create blank mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define torso region (ADJUST THESE VALUES FOR YOUR IMAGE)
    center_x = w // 2
    center_y_top = int(h * 0.28)     # Top of shirt (below neck)
    center_y_bottom = int(h * 0.60)  # Bottom of shirt (above waist)
    
    width_top = int(w * 0.35)        # Shoulder width
    width_bottom = int(w * 0.32)     # Waist width
    
    # Create trapezoid shape for torso
    points = np.array([
        [center_x - width_top, center_y_top],      # Top left
        [center_x + width_top, center_y_top],      # Top right
        [center_x + width_bottom, center_y_bottom], # Bottom right
        [center_x - width_bottom, center_y_bottom], # Bottom left
    ], dtype=np.int32)
    
    cv2.fillPoly(mask, [points], 255)
    
    # Smooth the mask
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    # Save
    Image.fromarray(mask).save(output_path)
    
    # Also create overlay for verification
    img_array = np.array(img)
    overlay = img_array.copy()
    overlay[mask > 50] = [255, 0, 0]
    result = cv2.addWeighted(img_array, 0.6, overlay, 0.4, 0)
    Image.fromarray(result).save(output_path.replace('.png', '_overlay.png'))
    
    print(f"✅ Perfect mask created: {output_path}")
    print(f"✅ Overlay saved: {output_path.replace('.png', '_overlay.png')}")
    print("\n💡 If mask is wrong, adjust these values in the code:")
    print(f"   • center_y_top: {h * 0.28:.0f} (top of shirt)")
    print(f"   • center_y_bottom: {h * 0.60:.0f} (bottom of shirt)")
    print(f"   • width_top: {w * 0.35:.0f} (shoulder width)")
    print(f"   • width_bottom: {w * 0.32:.0f} (waist width)")
    
    return mask

if __name__ == "__main__":
    # Create perfect manual mask
    create_perfect_torso_mask(
        r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
        "perfect_mask.png"
    )
    
    # Then use it in try-on:
    from improved_sam_vton import ImprovedSAMVTON
    vton = ImprovedSAMVTON()
    
    # Use the manually created mask
    from efficient_batch_tryon import EfficientBatchVTON
    vton_pipeline = EfficientBatchVTON(mixed_precision="fp16")
    
    result = vton_pipeline.single_try_on(
        person_image=r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
        garment_image=r"C:\Projects\tivora\vto_mvp\notebooks\tshirt02.png",
        mask="perfect_mask.png",  # Use the perfect manual mask
        num_inference_steps=100,
        guidance_scale=2.5,
        width=768,
        height=1024,
        output_path="result_with_perfect_mask.png"
    )