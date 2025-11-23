from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

embedding_model = None
gemini_configured = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCss7mONolh071ibtAkyqK7sI-MjgMJgdw")

data_path = os.path.join(os.path.dirname(__file__), 'matched_data.json')
with open(data_path, 'r', encoding='utf-8') as f:
    GARMENTS = json.load(f)

IMAGES_DIR = r'C:\Users\Shiven\Downloads\archive (9)\data'
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

garment_embeddings = None

def hex_to_rgb(hex_code: str) -> tuple:
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def detect_undertone(hex_code: str) -> str:
    r, g, b = hex_to_rgb(hex_code)
    
    if r > g and r > b:
        if (r - g) > 15:
            return "warm"
        else:
            return "neutral"
    elif g > r and g > b:
        return "cool"
    elif b > g:
        return "cool"
    else:
        return "neutral"


def get_color_palette(undertone: str) -> dict:
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
    desc_lower = garment_desc.lower()
    match_count = sum(1 for color in palette["colors"] if color in desc_lower)
    avoid_count = sum(1 for color in palette.get("avoid", []) if color in desc_lower)
    score = (match_count * 0.3) - (avoid_count * 0.5)
    return max(0, min(1, score + 0.5))

def expand_query_with_gemini(base_query: str, gender: str) -> str:
    global gemini_configured
    
    if not GEMINI_API_KEY:
        return base_query
    
    try:
        if not gemini_configured:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_configured = True
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Given this fashion search query: "{base_query}"
For a {gender} customer, expand this with 5-8 relevant fashion keywords.
Return ONLY keywords as comma-separated, no explanations.

Query: {base_query}
Keywords:"""
        
        response = model.generate_content(prompt)
        expanded = response.text.strip()
        return f"{base_query} {expanded}"
        
    except Exception as e:
        print(f"Gemini failed: {e}")
        return base_query

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def compute_garment_embeddings():
    global garment_embeddings
    
    if garment_embeddings is not None:
        return garment_embeddings
    
    pkl_path = Path(__file__).parent / "garment_embeddings.pkl"
    
    if pkl_path.exists():
        print(f"Loading embeddings from {pkl_path.name}...")
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                garment_embeddings = data['embeddings']
                print(f"Loaded {data['num_garments']} embeddings")
                return garment_embeddings
        except Exception as e:
            print(f"Failed to load pickle: {e}")
            print("Computing from scratch...")
    else:
        print(f"No embeddings found, computing...")
    
    model = get_embedding_model()
    texts = [f"{g['display_name']} {g['category']} {g['description']}" for g in GARMENTS]
    garment_embeddings = model.encode(texts, show_progress_bar=False)
    print(f"Computed {len(garment_embeddings)} embeddings")
    
    return garment_embeddings

def semantic_search(query: str, top_k: int = 30) -> List[tuple]:
    model = get_embedding_model()
    embeddings = compute_garment_embeddings()
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(idx, similarities[idx]) for idx in top_indices]

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
    undertone = detect_undertone(request.skin_tone_hex)
    palette = get_color_palette(undertone)
    
    color_keywords = " ".join(palette["colors"][:3])
    gender_category = "mens wear casual formal" if request.gender == "male" else "womens wear casual formal"
    base_query = f"{color_keywords} {gender_category} clothing fashion"
    
    expanded_query = expand_query_with_gemini(base_query, request.gender)
    
    print(f"Undertone: {undertone}")
    print(f"Query: {base_query}")
    
    search_results = semantic_search(expanded_query, top_k=30)
    
    candidates = []
    for idx, semantic_score in search_results:
        garment = GARMENTS[idx]
        color_score = color_match_score(garment['description'], palette)
        final_score = (semantic_score * 0.7) + (color_score * 0.3)
        
        candidates.append({
            "garment": garment,
            "score": final_score,
            "idx": idx
        })
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    top_results = candidates[:request.num_items]
    
    return [
        GarmentResponse(
            id=str(c['idx']),
            name=c['garment']['display_name'],
            image_url=f"http://localhost:8002/images/{os.path.basename(c['garment']['image'])}",
            category=c['garment']['category'],
            description=c['garment']['description'][:150],
            score=round(float(c['score']), 3),
            undertone_match=undertone
        )
        for c in top_results
    ]


if __name__ == "__main__":
    import uvicorn
    print("Starting Recommendation API...")
    compute_garment_embeddings()
    print("Server ready!")
    uvicorn.run(app, host="0.0.0.0", port=8002)
