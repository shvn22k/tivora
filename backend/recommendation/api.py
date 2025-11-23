"""
Recommendation API - Advanced Garment Recommendations with Color Theory & Semantic Search
Port: 8002
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
import json
import os
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

app = FastAPI(title="Recommendation API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading)
embedding_model = None
gemini_configured = False

# Configure Gemini (put your API key here or env var)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCss7mONolh071ibtAkyqK7sI-MjgMJgdw")  # Add your key here

# Load garment data
data_path = os.path.join(os.path.dirname(__file__), r'C:\Projects\tivora\backend\recommendation\matched_data.json')
with open(data_path, 'r', encoding='utf-8') as f:
    GARMENTS = json.load(f)

# Precompute embeddings cache
garment_embeddings = None


# ==================== COLOR THEORY ====================

def hex_to_rgb(hex_code: str) -> tuple:
    """Convert hex code to RGB tuple"""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def detect_undertone(hex_code: str) -> str:
    """Detect skin undertone from hex color"""
    r, g, b = hex_to_rgb(hex_code)
    
    # Simple undertone detection based on RGB ratios
    if r > g and r > b:
        if (r - g) > 15:
            return "warm"  # More red
        else:
            return "neutral"
    elif g > r and g > b:
        return "cool"  # More green/yellow (olive/cool)
    elif b > g:
        return "cool"  # More blue
    else:
        return "neutral"


def get_color_palette(undertone: str) -> dict:
    """Get color palette based on undertone"""
    palettes = {
        "warm": {
            "colors": ["warm browns", "terracotta", "olive green", "rust", "gold", "peach", "coral"],
            "avoid": ["icy blue", "pure white", "gray"]
        },
        "cool": {
            "colors": ["navy blue", "emerald", "purple", "burgundy", "silver", "icy pink", "true white"],
            "avoid": ["orange", "bright yellow", "warm browns"]
        },
        "neutral": {
            "colors": ["navy", "gray", "teal", "soft pink", "jade green", "dusty rose", "beige"],
            "avoid": []
        }
    }
    return palettes.get(undertone, palettes["neutral"])


def color_match_score(garment_desc: str, palette: dict) -> float:
    """Score garment based on color palette match"""
    desc_lower = garment_desc.lower()
    
    # Positive matches
    match_count = sum(1 for color in palette["colors"] if color in desc_lower)
    
    # Negative matches (avoid colors)
    avoid_count = sum(1 for color in palette.get("avoid", []) if color in desc_lower)
    
    # Calculate score (0-1 range)
    score = (match_count * 0.3) - (avoid_count * 0.5)
    return max(0, min(1, score + 0.5))  # Normalize to 0-1


# ==================== GEMINI QUERY EXPANSION ====================

def expand_query_with_gemini(base_query: str, gender: str) -> str:
    """Use Gemini to expand search query with fashion keywords"""
    global gemini_configured
    
    if not GEMINI_API_KEY:
        # Fallback if no API key
        return base_query
    
    try:
        if not gemini_configured:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_configured = True
        
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""Given this fashion search query: "{base_query}"
For a {gender} customer, expand this with 5-8 relevant fashion keywords and style descriptors.
Return ONLY the keywords as a comma-separated list, no explanations.

Example: "warm tones casual wear" → "warm, earthy, casual, comfortable, relaxed fit, cotton, everyday wear"

Query: {base_query}
Keywords:"""
        
        response = model.generate_content(prompt)
        expanded = response.text.strip()
        
        # Combine original + expanded
        return f"{base_query} {expanded}"
        
    except Exception as e:
        print(f"Gemini expansion failed: {e}, using base query")
        return base_query


# ==================== EMBEDDING & SIMILARITY ====================

def get_embedding_model():
    """Lazy load embedding model"""
    global embedding_model
    if embedding_model is None:
        print("Loading sentence-transformers model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model


def compute_garment_embeddings():
    """Load precomputed embeddings from pickle, or compute if not available"""
    global garment_embeddings
    
    if garment_embeddings is not None:
        return garment_embeddings
    
    # Try loading from pickle first
    pkl_path = Path(__file__).parent / "garment_embeddings.pkl"
    
    if pkl_path.exists():
        print(f"📦 Loading precomputed embeddings from {pkl_path.name}...")
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                garment_embeddings = data['embeddings']
                print(f"✅ Loaded {data['num_garments']} embeddings (model: {data['model_name']})")
                return garment_embeddings
        except Exception as e:
            print(f"⚠️ Failed to load pickle: {e}")
            print("📊 Computing embeddings from scratch...")
    else:
        print(f"⚠️ No precomputed embeddings found at {pkl_path}")
        print("💡 Run 'python backend/recommendation/precompute_embeddings.py' to speed up startup")
        print("📊 Computing embeddings now (this will take ~30 seconds)...")
    
    # Fallback: compute embeddings
    model = get_embedding_model()
    
    texts = [
        f"{g['display_name']} {g['category']} {g['description']}"
        for g in GARMENTS
    ]
    
    garment_embeddings = model.encode(texts, show_progress_bar=False)
    print(f"✅ Computed {len(garment_embeddings)} embeddings")
    
    return garment_embeddings


def semantic_search(query: str, top_k: int = 30) -> List[tuple]:
    """Perform semantic search on garments"""
    model = get_embedding_model()
    embeddings = compute_garment_embeddings()
    
    # Encode query
    query_embedding = model.encode([query])
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return (index, similarity_score) tuples
    return [(idx, similarities[idx]) for idx in top_indices]


# ==================== MAIN RECOMMENDATION PIPELINE ====================

class RecommendRequest(BaseModel):
    skin_tone_hex: str  # e.g., "#C5966C"
    gender: Literal["male", "female"]
    num_items: int = 15


class GarmentResponse(BaseModel):
    id: str
    name: str
    image_url: str
    category: str
    description: str
    score: float
    undertone_match: Optional[str] = None


@app.get("/")
def health_check():
    return {
        "service": "Recommendation API v2.0", 
        "port": 8002, 
        "total_garments": len(GARMENTS),
        "features": ["color_theory", "semantic_search", "gemini_expansion"]
    }


@app.post("/recommend", response_model=List[GarmentResponse])
def get_recommendations(request: RecommendRequest):
    """
    Advanced recommendation pipeline:
    1. Hex → Undertone
    2. Build base query
    3. Gemini expansion
    4. Semantic search (top 30)
    5. Color-theory re-ranking
    6. Return top 15
    """
    
    # Step 1: Detect undertone
    undertone = detect_undertone(request.skin_tone_hex)
    palette = get_color_palette(undertone)
    
    # Step 2: Build base query
    color_keywords = " ".join(palette["colors"][:3])
    gender_category = "mens wear casual formal" if request.gender == "male" else "womens wear casual formal"
    base_query = f"{color_keywords} {gender_category} clothing fashion"
    
    # Step 3: Gemini query expansion
    expanded_query = expand_query_with_gemini(base_query, request.gender)
    
    print(f"🎨 Undertone: {undertone}")
    print(f"🔍 Base Query: {base_query}")
    print(f"✨ Expanded Query: {expanded_query}")
    
    # Step 4: Semantic search (top 30)
    search_results = semantic_search(expanded_query, top_k=30)
    
    # Step 5: Apply color-theory re-ranking
    candidates = []
    for idx, semantic_score in search_results:
        garment = GARMENTS[idx]
        
        # Calculate color match score
        color_score = color_match_score(garment['description'], palette)
        
        # Combined score (70% semantic, 30% color theory)
        final_score = (semantic_score * 0.7) + (color_score * 0.3)
        
        candidates.append({
            "garment": garment,
            "score": final_score,
            "idx": idx
        })
    
    # Sort by final score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Step 6: Return top 15
    top_results = candidates[:request.num_items]
    
    return [
        GarmentResponse(
            id=str(c['idx']),
            name=c['garment']['display_name'],
            image_url=c['garment']['image'],
            category=c['garment']['category'],
            description=c['garment']['description'][:150],
            score=round(float(c['score']), 3),
            undertone_match=undertone
        )
        for c in top_results
    ]


if __name__ == "__main__":
    import uvicorn
    
    # Load embeddings on startup
    print("🚀 Starting Recommendation API v2.0...")
    compute_garment_embeddings()
    print("✅ Server ready!")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
