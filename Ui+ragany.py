import streamlit as st
import asyncio
import tempfile
import hashlib
import json
from pathlib import Path

# ---- RAGAnything imports (requires raganything>=0.1.31, lightrag>=0.4.2) ----
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# ---- Ollama config ----
OLLAMA_BASE = "http://localhost:11434/v1"
TEXT_MODEL = "gpt-oss:20b"
VISION_MODEL = "llava:34b"
EMBED_MODEL = "nomic-embed-text:latest"

WORKING_DIR = "./rag_storage_ollama"
REGISTRY = Path(WORKING_DIR) / "processed_files.json"

# ------------------ Helper functions ------------------

def file_hash(path: Path):
    """Calculate SHA256 hash for detecting changes"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_registry():
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {}

def save_registry(reg):
    REGISTRY.write_text(json.dumps(reg, indent=2))

# ------------------ Ollama Wrappers ------------------

async def ollama_text_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    return await openai_complete_if_cache(
        model=TEXT_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=OLLAMA_BASE,
        api_key="ollama",
        **kwargs,
    )

async def ollama_vision_complete(prompt, system_prompt=None, history_messages=None,
                                 image_base64=None, messages=None, **kwargs):

    # if direct messages provided
    if messages:
        return await openai_complete_if_cache(
            model=VISION_MODEL,
            prompt="",
            messages=messages,
            base_url=OLLAMA_BASE,
            api_key="ollama",
            **kwargs,
        )

    # classic image input to VLM
    multimodal_msg = [
        {"role": "system", "content": system_prompt or "You are a helpful visual AI assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        ]},
    ]

    return await openai_complete_if_cache(
        model=VISION_MODEL,
        messages=multimodal_msg,
        base_url=OLLAMA_BASE,
        api_key="ollama",
        **kwargs,
    )

async def ollama_embed_async(texts):
    emb = await openai_embed(
        texts=texts,
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE,
        api_key="ollama",
    )
    return emb.tolist()

def embedding_func_factory():
    return EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=ollama_embed_async,
    )

# ------------------ RAGAnything Object ------------------

@st.cache_resource
def init_rag():
    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=ollama_text_complete,
        vision_model_func=ollama_vision_complete,
        embedding_func=embedding_func_factory(),
    )

    # Fix compatibility
    async def _noop_mark(*args, **kwargs):
        return None
    rag._mark_multimodal_processing_complete = _noop_mark

    return rag

rag = init_rag()

# ------------------ Streamlit UI ------------------

st.title("📚 RAGAnything + Ollama (Multimodal)")

tab1, tab2 = st.tabs(["📤 Upload & Update Knowledge", "🔍 Query Knowledge Base"])

# ------------------ UPLOAD TAB ------------------

with tab1:
    st.subheader("Upload new documents (PDF, PNG, JPG, TXT, DOCX, etc.)")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "png", "jpg", "jpeg", "txt", "md", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        reg = load_registry()
        new_files = []

        for file in uploaded_files:
            # Save to temp folder for processing
            suffix = Path(file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getvalue())
                tmp_path = Path(tmp.name)

            h = file_hash(tmp_path)

            # Check if we already have this file
            if reg.get(file.name) == h:
                st.info(f"⚡ Already processed (skipped): {file.name}")
                continue

            # Process the file
            with st.spinner(f"Processing {file.name} ..."):
                try:
                    asyncio.run(rag.process_document_complete(
                        file_path=str(tmp_path),
                        output_dir="./output",
                        parse_method="auto",
                        display_stats=False
                    ))
                    new_files.append(file.name)
                    reg[file.name] = h
                except Exception as e:
                    st.error(f"Failed processing {file.name}: {e}")

        save_registry(reg)

        if new_files:
            st.success(f"Added to knowledge base: {', '.join(new_files)}")

# ------------------ QUERY TAB ------------------

with tab2:
    st.subheader("Ask questions about your documents")

    query = st.text_input("Enter your question")

    if st.button("Ask") and query.strip():
        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(rag.aquery(query, mode="hybrid"))
                st.write("### Answer")
                st.write(result)
            except Exception as e:
                st.error(f"Query failed: {e}")
