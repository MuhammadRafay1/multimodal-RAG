#!/usr/bin/env python3
"""
task1_pipeline.py

Complete pipeline for:
1) Data extraction & preprocessing from PDFs (text blocks + embedded images)
2) Text & image embeddings (sentence-transformers / CLIP)
3) Simple semantic search (cosine similarity) and retrieval API.

Assumptions:
- Place your PDFs in ./data/ (names like "1. Annual Report 2023-24.pdf", "2. financials.pdf", "3. FYP-Handbook-2023.pdf")
- This script will create ./output/ for extracted assets and indexes.

Usage:
    python task1_pipeline.py

Notes:
- First run may download models (internet required).
- If you want OCR on images, install tesseract separately and optionally integrate.
"""

import os
import io
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import nltk
import faiss
import cv2

# --------------------------
# CONFIG
# --------------------------
DATA_DIR = Path("Data")       # where your PDFs are stored
OUTPUT_DIR = Path("output")   # will hold extracted pages, images, chunks, embeddings, index files
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Chunking parameters
CHUNK_MAX_CHARS = 900   # target chunk size in characters
CHUNK_OVERLAP = 200     # overlap between chunks

# Models (names from sentence-transformers)
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"        # lightweight & good for semantic search
IMAGE_EMBED_MODEL = "clip-ViT-B-32"          # CLIP visual model via sentence-transformers

# Embedding filenames
TEXT_EMB_FILE = OUTPUT_DIR / "text_embeddings.npy"
META_FILE = OUTPUT_DIR / "chunks_metadata.parquet"
IMAGE_EMB_FILE = OUTPUT_DIR / "image_embeddings.npy"

# Retrieval settings
DEFAULT_TOPK = 5

# ensure nltk punkt

nltk.download('punkt')
nltk.download('punkt_tab')

# --------------------------
# Dataclasses
# --------------------------
@dataclass
class Chunk:
    id: str
    doc_name: str
    page_num: int
    chunk_type: str  # 'text' or 'figure' or 'table'
    text: str  # textual content or caption/ocr summary
    source_image: str  # if figure, path to image else "" 
    char_start: int
    char_end: int

# --------------------------
# Utilities: PDF parsing & chunking
# --------------------------

def list_pdfs(data_dir: Path) -> List[Path]:
    pdfs = sorted([p for p in data_dir.glob("*.pdf")])
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir.resolve()}. Put files there.")
    print(f"Found {len(pdfs)} PDFs.")
    return pdfs

def safe_load_image(path):
    try:
        img = Image.open(path)
        img = img.convert("RGB")

        # Sometimes PDFs generate garbage sizes like 1x1 or 2x2
        if img.width < 10 or img.height < 10:
            print(f"[WARNING] Image too small, skipping: {path}")
            return None

        return img
    except Exception as e:
        print(f"[WARNING] Skipped corrupted image: {path} ({e})")
        return None



def extract_pdf_to_pages(pdf_path: Path, out_dir: Path) -> List[Dict]:
    """
    For each page, extract:
      - page_text (in reading order using get_text("blocks"))
      - render page as PNG (for UI / reference)
      - extract embedded images (if any) and save them
    Returns a list of dicts with page info.
    """
    out_dir = out_dir / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages_info = []
    for pnum in range(len(doc)):
        page = doc[pnum]
        # text extraction: use blocks to preserve block ordering and simple grouping
        blocks = page.get_text("blocks")  # list of tuples (x0, y0, x1, y1, "text", block_no)
        blocks_sorted = sorted(blocks, key=lambda b: (int(b[1]), int(b[0])))  # sort by y then x
        page_text = "\n".join([b[4].strip() for b in blocks_sorted if b[4].strip() != ""])
        # save page image (rasterized)
        pix = page.get_pixmap(dpi=150)
        page_img_name = out_dir / f"page_{pnum+1:03d}.png"
        pix.save(str(page_img_name))
        # extract embedded images (if any)
        images = []
        xrefs = page.get_images(full=True)
        for img_index, img in enumerate(xrefs):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_name = out_dir / f"page{pnum+1:03d}_img{img_index+1}.{img_ext}"
            with open(img_name, "wb") as f:
                f.write(image_bytes)
            images.append(str(img_name))
        pages_info.append({
            "doc_name": pdf_path.name,
            "page_num": pnum+1,
            "page_text": page_text,
            "page_image": str(page_img_name),
            "figures": images
        })
    doc.close()
    print(f"Extracted {len(pages_info)} pages and {sum(len(p['figures']) for p in pages_info)} images from {pdf_path.name}")
    return pages_info

def chunk_text(text: str, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP) -> List[Tuple[int,int,str]]:
    """
    Split `text` into overlapping chunks at sentence boundaries.
    Returns list of (start_char, end_char, chunk_text)
    """
    if not text or text.strip() == "":
        return []
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current = ""
    char_pos = 0
    start_char = 0
    for sent in sentences:
        if current == "":
            start_char = char_pos
        # ensure we keep additions
        if len(current) + len(sent) + 1 <= max_chars:
            if current:
                current = current + " " + sent
            else:
                current = sent
        else:
            end_char = start_char + len(current)
            chunks.append((start_char, end_char, current.strip()))
            # start new chunk with overlap: we will back up `overlap` chars from end to create overlap
            # approximate by joining last few sentences until overlap length reached
            # simpler: start new current with this sentence
            # but to keep overlap we can keep tail substring:
            tail = current[-overlap:] if overlap < len(current) else current
            current = (tail + " " + sent).strip()
            start_char = end_char - len(tail)  # approximate
        char_pos += len(sent) + 1  # account for punctuation/space
    # append remaining
    if current:
        end_char = start_char + len(current)
        chunks.append((start_char, end_char, current.strip()))
    # adjust chunk start/end to be within text length
    return chunks

# --------------------------
# Build chunks dataset
# --------------------------
def build_chunks_from_pdfs(data_dir: Path, output_dir: Path) -> List[Chunk]:
    pdfs = list_pdfs(data_dir)
    all_chunks: List[Chunk] = []
    for pdf in pdfs:
        pages = extract_pdf_to_pages(pdf, output_dir)
        for p in pages:
            text = p["page_text"]
            page_img = p["page_image"]
            # 1) chunk page text
            text_chunks = chunk_text(text)
            for i, (s, e, chunk_str) in enumerate(text_chunks):
                cid = f"{pdf.stem}_p{p['page_num']:03d}_t{i+1}"
                ch = Chunk(
                    id=cid,
                    doc_name=pdf.name,
                    page_num=p["page_num"],
                    chunk_type="text",
                    text=chunk_str,
                    source_image="",
                    char_start=s,
                    char_end=e
                )
                all_chunks.append(ch)
            # 2) each figure becomes a chunk (use caption guess: we don't have caption detection - so we include OCR placeholder)
            for f_idx, fig_path in enumerate(p["figures"]):
                cid = f"{pdf.stem}_p{p['page_num']:03d}_f{f_idx+1}"
                # naive caption attempt: search for short text near "Figure" in page text
                short_cap = ""
                # simple heuristic: find lines containing "Figure" or "Fig."
                for line in text.splitlines():
                    if "Figure" in line or "Fig." in line or "FIGURE" in line or "Fig " in line:
                        short_cap = short_cap + " " + line.strip()
                chunk_txt = (short_cap.strip() + " ") if short_cap.strip() != "" else ""
                chunk_txt += f"[Figure image at {fig_path}]"
                ch = Chunk(
                    id=cid,
                    doc_name=pdf.name,
                    page_num=p["page_num"],
                    chunk_type="figure",
                    text=chunk_txt,
                    source_image=str(fig_path),
                    char_start=0,
                    char_end=0
                )
                all_chunks.append(ch)
    # Save chunks to JSONL & parquet metadata
    meta_rows = []
    for c in all_chunks:
        meta_rows.append({
            "id": c.id,
            "doc_name": c.doc_name,
            "page_num": c.page_num,
            "chunk_type": c.chunk_type,
            "text": c.text,
            "source_image": c.source_image,
            "char_start": c.char_start,
            "char_end": c.char_end
        })
    df_meta = pd.DataFrame(meta_rows)
    df_meta.to_parquet(META_FILE, index=False)
    with open(OUTPUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Built {len(all_chunks)} chunks. Metadata written to {META_FILE}")
    return all_chunks

# --------------------------
# Embeddings
# --------------------------
def load_models(text_model_name=TEXT_EMBED_MODEL, image_model_name=IMAGE_EMBED_MODEL):
    print("Loading models (this may download weights on first run)...")
    text_model = SentenceTransformer(text_model_name)
    # image model (CLIP) can be used for images; many sentence-transformers CLIP models accept PIL images
    try:
        image_model = SentenceTransformer(image_model_name)
    except Exception as e:
        print("Could not load image model; falling back to using text embeddings for images (caption-only). Error:", e)
        image_model = None
    return text_model, image_model

def compute_text_embeddings(chunks: List[Chunk], text_model: SentenceTransformer, batch_size=64):
    texts = [c.text for c in chunks]
    # For figure chunks that have only "[Figure image at ...]" we might want to replace with a placeholder caption,
    # but we'll still generate embeddings for that text representation.
    print(f"Computing text embeddings for {len(texts)} chunks...")
    embeddings = text_model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    np.save(TEXT_EMB_FILE, embeddings)
    print(f"Saved text embeddings to {TEXT_EMB_FILE}")
    return embeddings

def compute_image_embeddings(chunks: List[Chunk], image_model: SentenceTransformer, batch_size=32):
    # gather image paths
    img_paths = []
    img_indices = []  # map index in embeddings to chunk index
    for idx, c in enumerate(chunks):
        if c.source_image and os.path.exists(c.source_image):
            img_paths.append(c.source_image)
            img_indices.append(idx)
    if not img_paths:
        print("No figure images found for image embeddings.")
        return None, img_indices
    print(f"Computing image embeddings for {len(img_paths)} images...")
    # load images as PIL
    pil_images = []
    valid_indices = []
    for i, p in enumerate(img_paths):
        img = safe_load_image(p)
        if img is not None:
            pil_images.append(img)
            valid_indices.append(i)
        else:
            print(f"Skipping image {p}")

    img_embs = image_model.encode(
        pil_images,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    # allocate full image_embeddings aligned to chunks (None where no image)
    full_img_embeddings = np.zeros((len(chunks), img_embs.shape[1]), dtype=np.float32)  
    for emb_idx, img_list_idx in enumerate(valid_indices):
        chunk_idx = img_indices[img_list_idx]
        full_img_embeddings[chunk_idx] = img_embs[emb_idx]
    np.save(IMAGE_EMB_FILE, full_img_embeddings)
    print(f"Saved image embeddings to {IMAGE_EMB_FILE}")
    return full_img_embeddings, img_indices

# --------------------------
# Index (FAISS) + retrieval
# --------------------------
def build_faiss_index(embeddings: np.ndarray, index_path: Path = OUTPUT_DIR / "faiss.index"):
    d = embeddings.shape[1]
    # use IndexFlatIP and store normalized embeddings for cosine similarity via inner product
    index = faiss.IndexFlatIP(d)
    # already normalized in our embedding pipeline, so we can directly add
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(index_path))
    print(f"FAISS index built with {index.ntotal} vectors and saved to {index_path}")
    return index

def load_faiss_index(index_path: Path):
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file {index_path} not found.")
    index = faiss.read_index(str(index_path))
    return index

def cosine_search(query_emb: np.ndarray, indexed_embs: np.ndarray, top_k=DEFAULT_TOPK):
    """
    Simple linear cosine search (works fine for small collections).
    If embeddings already normalized, cosine similarity = dot product.
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    sims = (indexed_embs @ query_emb.T).squeeze(-1)  # shape (N,)
    # argsort descending
    topk_idx = np.argsort(-sims)[:top_k]
    topk_sims = sims[topk_idx]
    return topk_idx, topk_sims

# --------------------------
# High-level functions to call
# --------------------------
def prepare_pipeline():
    # 1) parse & chunk
    chunks = build_chunks_from_pdfs(DATA_DIR, OUTPUT_DIR)
    # 2) load models
    text_model, image_model = load_models()
    # 3) compute embeddings
    text_embeddings = compute_text_embeddings(chunks, text_model)
    # 4) image embeddings (optional)
    if image_model:
        image_embeddings, img_indices = compute_image_embeddings(chunks, image_model)
    else:
        image_embeddings, img_indices = None, []
    # 5) build faiss index for text embeddings
    faiss_index = build_faiss_index(text_embeddings)
    # store chunks to disk for later retrieval use
    with open(OUTPUT_DIR / "chunks_meta.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
    print("Pipeline prepared and artifacts saved to output/")
    return chunks, text_embeddings, image_embeddings, faiss_index, text_model, image_model

def semantic_search_text(query: str, text_model: SentenceTransformer, chunks: List[Chunk], text_embeddings: np.ndarray, top_k=5, hybrid_with_image=False, image_embeddings=None):
    q_emb = text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    # text-only similarity
    idxs, sims = cosine_search(q_emb, text_embeddings, top_k=top_k)
    results = []
    for i, s in zip(idxs, sims):
        results.append({"chunk": chunks[i], "score": float(s), "mode": "text"})
    # if hybrid enabled & image embeddings present: also search image embeddings and merge results
    if hybrid_with_image and image_embeddings is not None:
        idxs_i, sims_i = cosine_search(q_emb, image_embeddings, top_k=top_k)
        for i, s in zip(idxs_i, sims_i):
            results.append({"chunk": chunks[i], "score": float(s), "mode": "image"})
        # merge results by score and unique chunk id
        results = sorted(results, key=lambda r: -r["score"])
        unique = {}
        merged = []
        for r in results:
            if r["chunk"].id not in unique:
                unique[r["chunk"].id] = True
                merged.append(r)
        results = merged[:top_k]
    return results

def semantic_search_image(query_image_path: str, image_model: SentenceTransformer, chunks: List[Chunk], image_embeddings: np.ndarray, text_model=None, text_embeddings=None, top_k=5, hybrid_with_text=False):
    # embed the input image
    pil = Image.open(query_image_path).convert("RGB")
    q_emb = image_model.encode([pil], convert_to_numpy=True, normalize_embeddings=True)[0]
    idxs, sims = cosine_search(q_emb, image_embeddings, top_k=top_k)
    results = [{"chunk": chunks[i], "score": float(s), "mode": "image"} for i, s in zip(idxs, sims)]
    if hybrid_with_text and text_model is not None and text_embeddings is not None:
        # fallback: create caption of input image? (not done here). Instead, use image embedding to query text embeddings via cross-modal mapping (works when both models live in same space)
        idxs_t, sims_t = cosine_search(q_emb, text_embeddings, top_k=top_k)
        for i, s in zip(idxs_t, sims_t):
            results.append({"chunk": chunks[i], "score": float(s), "mode": "text"})
        results = sorted(results, key=lambda r: -r["score"])
        unique = {}
        merged = []
        for r in results:
            if r["chunk"].id not in unique:
                unique[r["chunk"].id] = True
                merged.append(r)
        results = merged[:top_k]
    return results

def load_pipeline():
    # 1) load chunks
    with open(OUTPUT_DIR / "chunks_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunks = [Chunk(**m) for m in meta]

    # 2) load embeddings
    text_embeddings = np.load(TEXT_EMB_FILE)

    image_embeddings = None
    if IMAGE_EMB_FILE.exists():
        image_embeddings = np.load(IMAGE_EMB_FILE)

    # 3) load models
    text_model, image_model = load_models()

    # 4) load faiss index
    faiss_index = load_faiss_index(OUTPUT_DIR / "faiss.index")

    return chunks, text_embeddings, image_embeddings, faiss_index, text_model, image_model

# --------------------------
# Small demo usage
# --------------------------
def demo_interaction(chunks, text_embeddings, image_embeddings, text_model, image_model):
    print("\n=== DEMO: Text query ===")
    q = "Generative AI final year projects"
    res = semantic_search_text(q, text_model, chunks, text_embeddings, top_k=5, hybrid_with_image=False, image_embeddings=image_embeddings)
    for r in res:
        c = r["chunk"]
        print(f"- [{r['mode']}] {c.id} (doc={c.doc_name} page={c.page_num}) score={r['score']:.4f}")
        snippet = c.text[:400].replace("\n", " ")
        print("   snippet:", snippet)
    # If there are images in output folder, let's do an image query demo
    print("\n=== DEMO: Image query (if images exist) ===")
    # find first figure path if any
    fig_path = None
    for c in chunks:
        if c.chunk_type == "figure" and c.source_image and os.path.exists(c.source_image):
            fig_path = c.source_image
            break
    if fig_path and image_model is not None and image_embeddings is not None:
        res_img = semantic_search_image(fig_path, image_model, chunks, image_embeddings, text_model=text_model, text_embeddings=text_embeddings, top_k=5, hybrid_with_text=False)
        print(f"Using image: {fig_path}")
        for r in res_img:
            c = r["chunk"]
            print(f"- [{r['mode']}] {c.id} (doc={c.doc_name} page={c.page_num}) score={r['score']:.4f}")
            print("   source_image:", c.source_image)
    else:
        print("No figure image available to run image-to-doc demo (or image model missing).")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # Build pipeline (extract->chunks->embeddings->index)
    chunks, text_embeddings, image_embeddings, faiss_index, text_model, image_model = load_pipeline()
    # run demo queries
    demo_interaction(chunks, text_embeddings, image_embeddings, text_model, image_model)
    print("\nAll done. Artifacts saved in:", OUTPUT_DIR.resolve())
