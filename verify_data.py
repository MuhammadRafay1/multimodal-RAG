import sys
import os
import numpy as np
import json
import faiss
from pathlib import Path

# Add scripts/text-extraction to path to import Chunk class if needed, 
# but we can just load the json directly.

OUTPUT_DIR = Path("e:/assignments/genAI-A3/scripts/text-extraction/output")
TEXT_EMB_FILE = OUTPUT_DIR / "text_embeddings.npy"
IMAGE_EMB_FILE = OUTPUT_DIR / "image_embeddings.npy"
META_FILE = OUTPUT_DIR / "chunks_meta.json"
INDEX_FILE = OUTPUT_DIR / "faiss.index"

def verify():
    print(f"Checking {OUTPUT_DIR}...")
    
    if not TEXT_EMB_FILE.exists():
        print(f"MISSING: {TEXT_EMB_FILE}")
        return
    
    text_embs = np.load(TEXT_EMB_FILE)
    print(f"Text Embeddings shape: {text_embs.shape}")
    
    if IMAGE_EMB_FILE.exists():
        img_embs = np.load(IMAGE_EMB_FILE)
        print(f"Image Embeddings shape: {img_embs.shape}")
    else:
        print("Image embeddings not found (optional).")
        
    if not META_FILE.exists():
        print(f"MISSING: {META_FILE}")
        return
        
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"Metadata entries: {len(meta)}")
    
    if not INDEX_FILE.exists():
        print(f"MISSING: {INDEX_FILE}")
        return
        
    index = faiss.read_index(str(INDEX_FILE))
    print(f"FAISS Index ntotal: {index.ntotal}")
    
    assert len(meta) == text_embs.shape[0]
    assert index.ntotal == text_embs.shape[0]
    
    print("VERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    verify()
