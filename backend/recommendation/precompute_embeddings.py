import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

print("Loading garment data...")
data_path = Path(__file__).parent / "matched_data.json"
with open(data_path, 'r', encoding='utf-8') as f:
    GARMENTS = json.load(f)

print(f"Found {len(GARMENTS)} garments")

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Computing embeddings...")
texts = [
    f"{g['display_name']} {g['category']} {g['description']}"
    for g in GARMENTS
]

embeddings = model.encode(texts, show_progress_bar=True)

print(f"Computed {len(embeddings)} embeddings")
print(f"Shape: {embeddings.shape}")

output_path = Path(__file__).parent / "garment_embeddings.pkl"
with open(output_path, 'wb') as f:
    pickle.dump({
        'embeddings': embeddings,
        'num_garments': len(GARMENTS),
        'model_name': 'all-MiniLM-L6-v2'
    }, f)

print(f"Saved to: {output_path}")
print(f"Size: {output_path.stat().st_size / 1024:.2f} KB")
print("Done!")

