from rag_core import MultimodalRAG

def diagnose():
    print("Initializing RAG...")
    rag = MultimodalRAG()
    
    # Test queries that should have clear answers in the docs
    test_queries = [
        "can you give some FYPs regarding cyber security",
        "List the Final Year Projects related to AI",
        "What is the revenue growth?"
    ]
    
    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"QUERY: {q}")
        print(f"{'='*50}")
        
        # 1. Check Retrieval
        print("--- RETRIEVAL ---")
        results = rag.retrieve(q, top_k=3)
        for i, r in enumerate(results):
            print(f"Rank {i+1} (Score: {r['score']:.4f})")
            print(f"Source: {r['chunk']['doc_name']} (Page {r['chunk']['page_num']})")
            print(f"Text: {r['chunk']['text'][:200]}...") # Show first 200 chars
            print("-" * 20)
            
        # 2. Check Generation
        print("\n--- GENERATION ---")
        response = rag.generate_response(q, results)
        print(f"Response:\n{response}")

if __name__ == "__main__":
    diagnose()
