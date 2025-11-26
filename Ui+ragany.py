# streamlit_rag_ollama.py
import streamlit as st
import tempfile
from pathlib import Path
import hashlib
import json
import asyncio
import threading
import base64
import time
import os

# RAGAnything / LightRAG imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# -------------------- USER-CONFIG --------------------
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434/v1")
TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "gpt-oss:20b")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:34b")      # optional: VLM name
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

WORKING_DIR = Path(os.getenv("RAG_WORKDIR", "./rag_storage_ollama"))
REGISTRY_PATH = WORKING_DIR / "processed_files.json"
# -----------------------------------------------------

WORKING_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- GLOBAL ASYNC LOOP --------------------
# Start a single, persistent asyncio loop in a background thread
GLOBAL_LOOP = asyncio.new_event_loop()


def _start_global_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


_thread = threading.Thread(target=_start_global_loop, args=(GLOBAL_LOOP,), daemon=True)
_thread.start()


def run_async(coro, timeout=None):
    """Submit coroutine to the single global loop and return result (blocks current thread).
       If timeout is None, wait indefinitely."""
    future = asyncio.run_coroutine_threadsafe(coro, GLOBAL_LOOP)
    return future.result(timeout=timeout)


# -------------------- Ollama (OpenAI-compatible) wrappers --------------------
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
    # If messages provided (multimodal message format), call with messages
    if messages:
        return await openai_complete_if_cache(
            model=VISION_MODEL,
            prompt="",
            messages=messages,
            base_url=OLLAMA_BASE,
            api_key="ollama",
            **kwargs,
        )

    # classic image input
    multimodal_msg = [
        {"role": "system", "content": system_prompt or "You are a helpful visual assistant."},
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
    # ensure list-of-lists
    return emb.tolist()


def embedding_func_factory():
    return EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=ollama_embed_async,
    )


# -------------------- RAGAnything init (cached) --------------------
@st.cache_resource(show_spinner=False)
def init_rag():
    # Configure RAGAnything
    config = RAGAnythingConfig(
        working_dir=str(WORKING_DIR),
        parser="mineru",          # mineru or docling
        parse_method="auto",      # auto/ocr/txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Create RAGAnything instance with our Ollama wrapper functions
    rag = RAGAnything(
        config=config,
        llm_model_func=ollama_text_complete,
        vision_model_func=ollama_vision_complete,
        embedding_func=embedding_func_factory(),
    )

    # Compatibility workaround: RAGAnything may call internal mark functions; noop them
    async def _noop_mark(*args, **kwargs):
        return None

    # replace if exists
    rag._mark_multimodal_processing_complete = _noop_mark
    return rag


rag = init_rag()

# -------------------- helper functions --------------------
def file_hash(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_registry():
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_registry(reg):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def save_uploaded_tmp(uploaded_file):
    # write to a persistent tmp file and return Path
    suffix = Path(uploaded_file.name).suffix
    tmp_dir = WORKING_DIR / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # ensure unique name: include timestamp
    tpath = tmp_dir / f"{int(time.time()*1000)}_{uploaded_file.name}"
    with open(tpath, "wb") as f:
        f.write(uploaded_file.getvalue())
    return tpath


def base64_from_path(path: Path):
    b = path.read_bytes()
    return base64.b64encode(b).decode("utf-8")


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="RAG-Anything + Ollama Chatbot", layout="wide")
st.title("📚 RAG-Anything + Ollama — Multimodal Chatbot (single-loop safe)")

# session state for chat history
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (user, assistant)

tab_upload, tab_chat = st.tabs(["📤 Upload & Update Knowledge", "💬 Chat / Query"])

with tab_upload:
    st.header("Upload files to add to the knowledge base")
    st.markdown("Supported types: `pdf`, `png`, `jpg`, `jpeg`, `txt`, `md`, `docx` (others may work depending on extras).")
    uploaded = st.file_uploader("Choose files", accept_multiple_files=True,
                                type=["pdf", "png", "jpg", "jpeg", "txt", "md", "docx"])
    if uploaded:
        registry = load_registry()
        new_added = []
        for f in uploaded:
            tmp_path = save_uploaded_tmp(f)
            h = file_hash(tmp_path)
            # If same name and same hash -> skip
            key = f.name
            if registry.get(key) == h:
                st.info(f"Skipped (already processed): {f.name}")
                continue

            st.info(f"Processing: {f.name}")
            try:
                # process doc on global loop
                run_async(rag.process_document_complete(
                    file_path=str(tmp_path),
                    output_dir=str(WORKING_DIR / "output"),
                    parse_method="auto",
                    display_stats=True
                ), timeout=900)  # may take long for large PDFs; adjust timeout
                registry[key] = h
                save_registry(registry)
                new_added.append(f.name)
                st.success(f"Processed and added: {f.name}")
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")

        if new_added:
            st.success("New files indexed: " + ", ".join(new_added))

    st.markdown("---")
    st.write("Index status:")
    st.write(f"Working dir: `{WORKING_DIR}`")
    st.write(f"Registry entries: {len(load_registry())}")

with tab_chat:
    st.header("Ask questions about the uploaded documents")
    query = st.text_input("Your question", key="query_input")
    col1, col2 = st.columns([3, 1])
    with col2:
        st.write("Options")
        mode = st.selectbox("Query mode", options=["hybrid", "local", "global", "naive"], index=0)
        vlm_choice = st.checkbox("Enable VLM (vision) enhancement when images present", value=True)
        topk = st.slider("Top-k retrieved", 1, 10, 4)

    if st.button("Ask") and query.strip():
        st.session_state.chat.append(("user", query))
        placeholder = st.empty()
        with placeholder.container():
            st.write("Thinking... (this may take a few seconds)")
        try:
            # run aquery on global loop
            answer = run_async(rag.aquery(
                query,
                mode=mode,
                vlm_enhanced=vlm_choice,
                top_k=topk
            ), timeout=120)  # adjust timeout as needed

            st.session_state.chat.append(("assistant", answer))
            placeholder.empty()
            st.success("Answer retrieved")
        except Exception as e:
            placeholder.empty()
            st.error(f"Query failed: {e}")

    # render chat history
    st.markdown("### Conversation")
    for role, text in st.session_state.chat[::-1]:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

    # simple "clear" controls
    clear_col1, clear_col2 = st.columns(2)
    if clear_col1.button("Clear chat"):
        st.session_state.chat = []
    if clear_col2.button("Rebuild index (reprocess all files)"):
        # caution: reprocess all files (use with care)
        registry = {}
        save_registry(registry)
        st.info("Registry cleared — next upload or reprocess will re-index files.")
        # optionally trigger processing existing uploaded folder again
        upload_folder = WORKING_DIR / "uploads"
        if upload_folder.exists():
            files = sorted(upload_folder.iterdir())
            if files:
                st.info(f"Reprocessing {len(files)} files...")
                for fp in files:
                    try:
                        run_async(rag.process_document_complete(
                            file_path=str(fp),
                            output_dir=str(WORKING_DIR / "output"),
                            parse_method="auto",
                            display_stats=False
                        ), timeout=600)
                        registry[fp.name] = file_hash(fp)
                        save_registry(registry)
                        st.success(f"Processed {fp.name}")
                    except Exception as e:
                        st.error(f"Failed {fp.name}: {e}")

st.markdown("---")
st.caption("Notes: This app submits all async calls to one global background loop to avoid worker/event-loop mismatches. Adjust timeouts for large PDFs or slow OCR.")
