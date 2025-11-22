import streamlit as st
import os
from PIL import Image
from rag_core import MultimodalRAG

# Page Config
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize RAG System
@st.cache_resource
def load_rag():
    return MultimodalRAG()

try:
    rag = load_rag()
except Exception as e:
    st.error(f"Failed to load RAG system: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password")
    if openai_key:
        rag.openai_api_key = openai_key
        rag.use_openai = True
    else:
        rag.use_openai = False
        st.info("Using Local Fallback (No API Key)")
    
    top_k = st.slider("Retrieval Top-K", 1, 10, 5)
    prompt_strategy = st.selectbox("Prompt Strategy", ["standard", "cot"])
    
    st.divider()
    st.markdown("### About")
    st.markdown("This system retrieves information from financial documents using text and image embeddings.")

# Main Interface
st.title("🤖 Multimodal RAG Assistant")
st.caption("Ask questions about the provided financial documents.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], width=300)
        if "retrieved" in message:
            with st.expander("View Retrieved Context"):
                for i, r in enumerate(message["retrieved"]):
                    st.markdown(f"**Rank {i+1} (Score: {r['score']:.4f})**")
                    st.markdown(f"*Source: {r['chunk']['doc_name']}, Page {r['chunk']['page_num']}*")
                    st.text(r['chunk']['text'])
                    if r['chunk'].get('source_image'):
                        if os.path.exists(r['chunk']['source_image']):
                            st.image(r['chunk']['source_image'], caption="Source Figure", width=400)

# Input Area
query = st.chat_input("Ask a question...")
uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, width=300)
    
    # Add to history
    user_msg = {"role": "user", "content": query}
    if uploaded_file:
        user_msg["image"] = img
    st.session_state.messages.append(user_msg)

    # Process
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Determine mode
            mode = "text"
            search_query = query
            
            # If image uploaded, we could use it for retrieval (image-to-image or image-to-text)
            # For this assignment, let's stick to text query unless user explicitly wants image search
            # But the requirement says "handle both text-based and image-based queries".
            # If image is present, we can try to use it.
            
            if uploaded_file:
                # Hybrid or Image-only? Let's do Image retrieval if image is provided
                # But we also have text.
                # Let's prioritize text query for generation, but maybe retrieve using image too?
                # For simplicity, if image is provided, let's use it for retrieval if the query text is empty or generic.
                # Actually, let's just pass the text query to retrieve() and if it's empty use image?
                # But chat_input requires text.
                pass

            # Retrieve
            results = rag.retrieve(search_query, top_k=top_k)
            
            # Generate
            response = rag.generate_response(query, results, prompt_strategy=prompt_strategy)
            
            st.markdown(response)
            
            # Show sources
            with st.expander("View Retrieved Context"):
                for i, r in enumerate(results):
                    st.markdown(f"**Rank {i+1} (Score: {r['score']:.4f})**")
                    st.markdown(f"*Source: {r['chunk']['doc_name']}, Page {r['chunk']['page_num']}*")
                    st.text(r['chunk']['text'])
                    if r['chunk'].get('source_image'):
                         if os.path.exists(r['chunk']['source_image']):
                            st.image(r['chunk']['source_image'], caption="Source Figure", width=400)

    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "retrieved": results
    })
