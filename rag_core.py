import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
from PIL import Image
import pandas as pd

# Configuration
# Assuming the script is run from the root of the project
DATA_OUTPUT_DIR = Path("scripts/text-extraction/output")
TEXT_EMB_FILE = DATA_OUTPUT_DIR / "text_embeddings.npy"
IMAGE_EMB_FILE = DATA_OUTPUT_DIR / "image_embeddings.npy"
META_FILE = DATA_OUTPUT_DIR / "chunks_meta.json"
INDEX_FILE = DATA_OUTPUT_DIR / "faiss.index"

# Models
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "clip-ViT-B-32"

class MultimodalRAG:
    def __init__(self, use_openai: bool = False, openai_api_key: Optional[str] = None):
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        
        print("Loading RAG resources...")
        self._load_metadata()
        self._load_index()
        self._load_models()
        print("RAG resources loaded.")

    def _load_metadata(self):
        if not META_FILE.exists():
            raise FileNotFoundError(f"Metadata file not found at {META_FILE}")
        with open(META_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        # Create a lookup for quick access if needed, though list index matches faiss index
        self.text_embeddings = np.load(TEXT_EMB_FILE)
        if IMAGE_EMB_FILE.exists():
            self.image_embeddings = np.load(IMAGE_EMB_FILE)
        else:
            self.image_embeddings = None

    def _load_index(self):
        if not INDEX_FILE.exists():
            raise FileNotFoundError(f"Index file not found at {INDEX_FILE}")
        self.index = faiss.read_index(str(INDEX_FILE))

    def _load_models(self):
        self.text_model = SentenceTransformer(TEXT_MODEL_NAME)
        try:
            self.image_model = SentenceTransformer(IMAGE_MODEL_NAME)
        except Exception as e:
            print(f"Warning: Could not load image model {IMAGE_MODEL_NAME}: {e}")
            self.image_model = None
        
        # Load Cross-Encoder for Re-ranking
        print("Loading Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"Warning: Could not load Cross-Encoder: {e}")
            self.cross_encoder = None

        # Load Local LLM (Flan-T5-Large is a good balance for CPU RAG)
        print("Loading local LLM (google/flan-t5-large)...")
        try:
            from transformers import pipeline
            self.local_llm = pipeline("text2text-generation", model="google/flan-t5-large")
        except Exception as e:
            print(f"Warning: Could not load local LLM: {e}")
            self.local_llm = None

    def retrieve(self, query: Union[str, Image.Image], top_k: int = 5, mode: str = "text") -> List[Dict]:
        """
        Retrieve relevant chunks based on query.
        mode: 'text' (query is string), 'image' (query is PIL Image)
        """
        if mode == "text":
            if isinstance(query, str):
                query_emb = self.text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            else:
                raise ValueError("Query must be a string for text mode")
        elif mode == "image":
            if self.image_model and isinstance(query, Image.Image):
                query_emb = self.image_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            else:
                raise ValueError("Image model not loaded or invalid query type for image mode")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # FAISS search
        # Retrieve more candidates for re-ranking (e.g., 3x top_k)
        initial_k = top_k * 10 # Increase candidates to cast a wider net
        D, I = self.index.search(query_emb, initial_k)
        
        candidates = []
        for i, idx in enumerate(I[0]):
            if idx == -1: continue 
            chunk = self.chunks[idx]
            score = float(D[0][i])
            
            # Keyword Boosting: If query terms appear in chunk text or doc name, boost score
            # This helps when semantic similarity is weak but keyword match is strong (e.g. "FYP")
            if isinstance(query, str):
                query_lower = query.lower()
                text_lower = chunk['text'].lower()
                doc_lower = chunk['doc_name'].lower()
                
                # Boost for explicit document reference
                if "fyp" in query_lower and "fyp" in doc_lower:
                    score += 0.5
                
                # Boost for exact phrase match
                if query_lower in text_lower:
                    score += 0.3
                
                # Boost for key terms
                keywords = ["cyber security", "ai", "artificial intelligence", "blockchain", "iot"]
                for kw in keywords:
                    if kw in query_lower and kw in text_lower:
                        score += 0.2
                        
            candidates.append({
                "chunk": chunk,
                "score": score,
                "rank": i + 1
            })
            
        # Re-ranking
        if self.cross_encoder and mode == "text" and isinstance(query, str):
            # Prepare pairs for cross-encoder
            pairs = [[query, c["chunk"]["text"]] for c in candidates]
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Update scores and sort
            for i, c in enumerate(candidates):
                # Combine vector score (boosted) with cross-encoder score
                # Cross-encoder score is usually logits, so we can just use it directly or mix
                # Let's trust cross-encoder more but keep the boost influence if needed
                # Actually, standard practice is to just use cross-encoder score for sorting
                # BUT, if we boosted the *candidates* selection, we might have better chunks in the pool.
                # Here we are re-ranking the *already retrieved* candidates.
                # So the boosting above mainly helps if we sort candidates BEFORE re-ranking?
                # No, FAISS returns sorted by vector score.
                # We boosted the score in the loop above.
                # Now we re-rank.
                c["score"] = float(cross_scores[i])
                c["original_rank"] = c["rank"]
                
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
            
        # Return top_k
        final_results = candidates[:top_k]
        
        # Re-assign ranks
        for i, r in enumerate(final_results):
            r["rank"] = i + 1
            
        return final_results

    def generate_response(self, query: str, retrieved_chunks: List[Dict], prompt_strategy: str = "standard") -> str:
        """
        Generate a response using the retrieved chunks.
        """
        # Format context with clear separators
        context_text = ""
        for c in retrieved_chunks:
            context_text += f"--- Source: {c['chunk']['doc_name']} (Page {c['chunk']['page_num']}) ---\n{c['chunk']['text']}\n\n"
        
        system_prompt = "You are an expert financial analyst. Your goal is to answer the user's question accurately using ONLY the provided context. If the answer is not in the context, state that you cannot find the information."
        
        if prompt_strategy == "cot":
            user_prompt = f"""
            Context Information:
            {context_text}
            
            User Question: {query}
            
            Instructions:
            1. Analyze the context to find relevant information from files.
            2. Think step-by-step to connect the facts to the question.
            3. Formulate a concise answer based on your analysis.
            
            Answer:
            """
        else:
            user_prompt = f"""
            Context Information:
            {context_text}
            
            User Question: {query}
            
            Answer:
            """

        if self.use_openai and self.openai_api_key:
            return self._call_openai(system_prompt, user_prompt)
        else:
            return self._local_fallback(user_prompt)

    def _call_openai(self, system_prompt, user_prompt):
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # Or gpt-4
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
        except ImportError:
            return "Error: OpenAI library not installed."
        except Exception as e:
            return f"Error calling OpenAI: {e}"

    def _local_fallback(self, prompt):
        if self.local_llm:
            try:
                # Flan-T5 works best with shorter inputs, so we might need to truncate if context is huge
                # But let's try passing it all.
                # Note: Flan-T5 max length is typically 512 tokens. We need to be careful.
                # Let's truncate prompt to ~2000 chars to be safe-ish for input
                truncated_prompt = prompt[-2500:] if len(prompt) > 2500 else prompt
                
                response = self.local_llm(truncated_prompt, max_length=512, do_sample=False)
                return response[0]['generated_text']
            except Exception as e:
                return f"Error generating local response: {e}"
        else:
            return f"[LOCAL FALLBACK - NO LLM LOADED]\n\nBased on the context provided, here are the relevant snippets:\n\n{prompt[:1000]}...\n\n(To get a real generated answer, please configure an OpenAI API key or ensure the local LLM is loaded)"

if __name__ == "__main__":
    # Test run
    rag = MultimodalRAG()
    print("Testing retrieval...")
    results = rag.retrieve("What is the revenue growth?", top_k=3)
    for r in results:
        print(f"Rank {r['rank']} (Score {r['score']:.4f}): {r['chunk']['text'][:100]}...")
    
    print("\nTesting generation (fallback)...")
    ans = rag.generate_response("What is the revenue growth?", results)
    print(ans)
