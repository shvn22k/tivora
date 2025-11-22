"""
Working Replicate VTON with IDM-VTON model - DEMO VERSION
Install: pip install replicate
"""

import replicate
import os
import sys

# Hardcoded API token for demo
os.environ["REPLICATE_API_TOKEN"] = "lol"


def tryon_idm_vton(person_image_path, garment_image_path, garment_description="clothing item", category="upper_body", output_path="outputs/result.jpg"):
    """
    Using IDM-VTON model on Replicate
    
    Args:
        person_image_path: Path to person image
        garment_image_path: Path to garment image
        garment_description: Description of the garment
        category: "upper_body", "lower_body", or "dresses"
        output_path: Where to save the result
    """
    print("🚀 Running IDM-VTON via Replicate...")
    print(f"   Category: {category}")
    
    try:
        # Prepare input for IDM-VTON model
        input_data = {
            "garm_img": open(garment_image_path, "rb"),
            "human_img": open(person_image_path, "rb"),
            "garment_des": garment_description,
            "category": category,
            "crop": False,
            "seed": 42,
            "steps": 30,
            "force_dc": False,
            "mask_only": False
        }
        
        print("📤 Uploading images and running model...")
        output = replicate.run(
            "cuuupid/idm-vton:0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
            input=input_data
        )
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the output file to disk
        print("💾 Saving result...")
        with open(output_path, "wb") as file:
            file.write(output.read())
        
        print(f"   ✅ Saved: {output_path}")
        print(f"\n🎉 Success! Virtual try-on complete")
        return output_path

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("🎨 Virtual Try-On Demo")
    print("=" * 60)
    print()
    
    # Use default images from inputs if no arguments provided
    if len(sys.argv) < 3:
        person_img = "inputs/person02.jpg"
        garment_img = "inputs/coolpolo.jpg"
        garment_desc = "polo tshirt"
        garment_category = "upper_body"  # "upper_body", "lower_body", or "dresses"
        print(f"📷 Using default demo images:")
        print(f"  Person: {person_img}")
        print(f"  Garment: {garment_img}")
        print(f"  Description: {garment_desc}")
        print(f"  Category: {garment_category}\n")
    else:
        person_img = sys.argv[1]
        garment_img = sys.argv[2]
        garment_desc = sys.argv[3] if len(sys.argv) > 3 else "polo tshirt"
        garment_category = sys.argv[4] if len(sys.argv) > 4 else "upper_body"
    
    # Run the virtual try-on
    result = tryon_idm_vton(person_img, garment_img, garment_desc, garment_category)
    
    if result:
        print(f"\n✨ Result saved: {result}")
        print(f"💡 Open the result to see the virtual try-on!")
    else:
        print("\n❌ Virtual try-on failed. Please check input images.")

