import sys
import os

# Add scripts directory to path so we can import efficient_batch_tryon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from efficient_batch_tryon import EfficientBatchVTON
from PIL import Image

# Initialize the pipeline
print("Loading model...")
vton = EfficientBatchVTON(
    base_model_path="booksforcharlie/stable-diffusion-inpainting",
    resume_path="zhengchong/CatVTON",
    dataset_name="vitonhd",
    mixed_precision="fp16",
    device="cuda",
    compile_model=False,  # Keep False on Windows (Triton not supported)
)

# Single try-on
print("Running try-on...")
result = vton.single_try_on(
    person_image=r"C:\Projects\tivora\vto_mvp\notebooks\sample01.jpeg",
    garment_image=r"C:\Projects\tivora\vto_mvp\notebooks\tshirt02.png",
    mask=None,  # Auto-generates a simple mask
    num_inference_steps=50,  # More steps = better quality
    guidance_scale=2.5,
    width=384,
    height=512,
    seed=42,
)

# Save result
result.save("output_single.png")
print("✅ Result saved to output_single.png")

# Display result
result.show()