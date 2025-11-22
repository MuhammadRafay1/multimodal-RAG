import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import json

# Config
DATA_OUTPUT_DIR = Path("scripts/text-extraction/output")
TEXT_EMB_FILE = DATA_OUTPUT_DIR / "text_embeddings.npy"
IMAGE_EMB_FILE = DATA_OUTPUT_DIR / "image_embeddings.npy"
META_FILE = DATA_OUTPUT_DIR / "chunks_meta.json"
PLOT_FILE = Path("embedding_visualization.png")

def visualize_embeddings():
    print("Loading embeddings for visualization...")
    if not TEXT_EMB_FILE.exists():
        print("Text embeddings not found.")
        return

    text_embs = np.load(TEXT_EMB_FILE)
    
    # Load metadata to color code or label
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    labels = [m["chunk_type"] for m in meta]
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(text_embs)-1))
    reduced = tsne.fit_transform(text_embs)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced[indices, 0], reduced[indices, 1], label=label, alpha=0.6)
        
    plt.title("t-SNE Visualization of Text Embeddings")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_FILE)
    print(f"Visualization saved to {PLOT_FILE}")

if __name__ == "__main__":
    visualize_embeddings()
