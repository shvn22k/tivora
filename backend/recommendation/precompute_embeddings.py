"""
Precompute embeddings for all garments and save to pickle
Run this once: python backend/recommendation/precompute_embeddings.py
"""
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

print("🚀 Loading garment data...")
data_path = Path(__file__).parent / "matched_data.json"
with open(data_path, 'r', encoding='utf-8') as f:
    GARMENTS = json.load(f)

print(f"📊 Found {len(GARMENTS)} garments")

print("🤖 Loading sentence-transformers model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("⚡ Computing embeddings...")
texts = [
    f"{g['display_name']} {g['category']} {g['description']}"
    for g in GARMENTS
]

embeddings = model.encode(texts, show_progress_bar=True)

print(f"✅ Computed {len(embeddings)} embeddings")
print(f"📦 Embedding shape: {embeddings.shape}")

# Save to pickle
output_path = Path(__file__).parent / "garment_embeddings.pkl"
with open(output_path, 'wb') as f:
    pickle.dump({
        'embeddings': embeddings,
        'num_garments': len(GARMENTS),
        'model_name': 'all-MiniLM-L6-v2'
    }, f)

print(f"💾 Saved embeddings to: {output_path}")
print(f"📏 File size: {output_path.stat().st_size / 1024:.2f} KB")
print("✨ Done! Server will now load these embeddings instantly.")

