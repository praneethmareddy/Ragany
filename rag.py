# rag_ollama_multimodal.py
"""
Full multimodal RAG-Anything + Ollama integration (single-file).
Requirements (see README below for commands):
 - Ollama running: `ollama serve`
 - Models pulled in Ollama:
    ollama pull gpt-oss:20b
    ollama pull llava:34b
    ollama pull nomic-embed-text:latest
 - Python packages: raganything==0.1.31, lightrag==0.4.2
 - MinerU installed for parsing (recommended)
 - LibreOffice (for office docs) if you use .docx/.pptx/.xlsx
"""

import os
import sys
import asyncio
import base64
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# ---- Environment defaults (override with .env or set here) ----
os.environ.setdefault("LLM_BINDING", "ollama")
os.environ.setdefault("LLM_BINDING_HOST", "http://localhost:11434")  # Ollama base
os.environ.setdefault("LLM_MODEL", "gpt-oss:20b")  # text model
os.environ.setdefault("VISION_MODEL", "llava:34b")  # vision model (LLaVA)
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text:latest")  # embedding model
os.environ.setdefault("WORKING_DIR", "./rag_storage_ollama")
os.environ.setdefault("DOCS_DIR", "./docs")  # folder to ingest
os.environ.setdefault("MAX_CONCURRENT_FILES", "3")

load_dotenv()  # optional .env support

OLLAMA_BASE = os.getenv("LLM_BINDING_HOST").rstrip("/") + "/v1"
TEXT_MODEL = os.getenv("LLM_MODEL")
VISION_MODEL = os.getenv("VISION_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
WORKING_DIR = os.getenv("WORKING_DIR")
DOCS_DIR = os.getenv("DOCS_DIR")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_FILES", "3"))

# ---- Imports from RAG-Anything / LightRAG ----
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
except Exception as e:
    print("Missing required libraries. Install with:\n"
          "pip install --no-cache-dir lightrag==0.4.2 raganything==0.1.31\n")
    raise

# ---- Helper: Ollama OpenAI-compatible wrappers (async) ----
async def ollama_text_complete(prompt: str,
                               system_prompt: Optional[str] = None,
                               history_messages: Optional[List[Dict]] = None,
                               **kwargs) -> str:
    """
    Use openai_complete_if_cache wrapper (compatible with Ollama's OpenAI style API)
    Note: openai_complete_if_cache expects base_url and api_key parameters.
    """
    return await openai_complete_if_cache(
        model=TEXT_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=OLLAMA_BASE,
        api_key="ollama",  # Ollama often requires no key; openai wrapper expects something
        **kwargs,
    )

async def ollama_embed_async(texts: List[str]) -> List[List[float]]:
    embeds = await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE,
        api_key="ollama",
    )
    # openai_embed may return numpy array or list-like; convert to python list
    try:
        return embeds.tolist()
    except Exception:
        return embeddings  # if already a list

# ---- Vision model wrapper (for VLM-enhanced queries)
# We'll send images as base64 data URIs in messages to the vision model.
async def ollama_vision_complete(prompt: str,
                                 system_prompt: Optional[str] = None,
                                 history_messages: Optional[List[Dict]] = None,
                                 image_base64: Optional[str] = None,
                                 messages: Optional[List[Dict]] = None,
                                 **kwargs) -> str:
    """
    If `messages` is provided (OpenAI-style multimodal chat), forward it.
    Else if image_base64 provided, build a messages list with data URI.
    """
    if messages:
        # pass through directly
        return await openai_complete_if_cache(
            model=VISION_MODEL,
            prompt="",
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            messages=messages,
            base_url=OLLAMA_BASE,
            api_key="ollama",
            **kwargs,
        )

    # build a simple multimodal message
    multimodal_msg = [
        {"role": "system", "content": system_prompt or "You are a helpful vision+language assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ]
    return await openai_complete_if_cache(
        model=VISION_MODEL,
        prompt="",
        system_prompt=None,
        history_messages=[],
        messages=multimodal_msg,
        base_url=OLLAMA_BASE,
        api_key="ollama",
        **kwargs,
    )

# ---- EmbeddingFunc factory for RAG-Anything (LightRAG compatible) ----
def embedding_func_factory():
    return EmbeddingFunc(
        embedding_dim=768,        # nomic-embed-text v1.5 uses 768 dims; adjust if different
        max_token_size=8192,
        func=ollama_embed_async,
    )

# ---- Main Integration Class ----
class OllamaRAGMultimodal:
    def __init__(self,
                 working_dir: str = WORKING_DIR,
                 docs_dir: str = DOCS_DIR,
                 max_concurrent: int = MAX_CONCURRENT):
        self.config = RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru",  # mineru recommended for PDFs/tables/equations
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        self.docs_dir = Path(docs_dir)
        self.rag: Optional[RAGAnything] = None
        self.max_concurrent = max_concurrent

    async def test_ollama(self) -> bool:
        """Quick check that Ollama is up and the models exist."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"{OLLAMA_BASE}/models") as r:
                    if r.status != 200:
                        print("Ollama responded with non-200:", r.status)
                        return False
                    data = await r.json()
                    available = [m.get("id") or m.get("name") for m in data.get("data", [])]
                    print("Ollama models available (sample):", available[:10])
                    if TEXT_MODEL not in available and TEXT_MODEL.split(":")[0] not in available:
                        print(f"Warning: text model {TEXT_MODEL} not listed. It may still work if ollama pulls by tag.")
                    if VISION_MODEL not in available and VISION_MODEL.split(":")[0] not in available:
                        print(f"Warning: vision model {VISION_MODEL} not listed.")
                    return True
        except Exception as e:
            print("Failed to contact Ollama at", OLLAMA_BASE, ":", e)
            return False

    async def initialize(self) -> bool:
        """Initialize RAGAnything with Ollama adapters."""
        try:
            print("Initializing RAG-Anything (multimodal) with Ollama...")
            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=ollama_text_complete,
                vision_model_func=ollama_vision_complete,
                embedding_func=embedding_func_factory(),
            )
            # compatibility workaround for some LightRAG versions
            async def _noop_mark(*args, **kwargs):
                return None
            self.rag._mark_multimodal_processing_complete = _noop_mark
            print("RAG initialized; working dir:", self.config.working_dir)
            return True
        except Exception as e:
            print("Failed to initialize RAGAnything:", e)
            return False

    async def process_single_file(self, file_path: Path):
        """Process a single document using RAG-Anything's API (process_document_complete)."""
        if not self.rag:
            raise RuntimeError("RAG not initialized")
        try:
            print(f"Processing {file_path} ...")
            await self.rag.process_document_complete(
                file_path=str(file_path),
                output_dir=os.path.join(self.config.working_dir, "output"),
                parse_method="auto",
                display_stats=True,
            )
            print(f"Completed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    async def ingest_folder(self, folder: Path):
        """Ingest all supported files in a folder, concurrently (bounded)."""
        files = [p for p in folder.rglob("*") if p.suffix.lower() in
                 {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md", ".png", ".jpg", ".jpeg"}]
        if not files:
            print("No supported files found in", folder)
            return

        sem = asyncio.Semaphore(self.max_concurrent)
        async def worker(p):
            async with sem:
                await self.process_single_file(p)

        print(f"Found {len(files)} files; processing with concurrency={self.max_concurrent}")
        await asyncio.gather(*(worker(p) for p in files))

    async def insert_sample_content(self):
        """Add a small sample text content using insert_content_list (async)."""
        if not self.rag:
            raise RuntimeError("RAG not initialized")
        print("Inserting small sample content for testing...")
        content_list = [
            {"type": "text",
             "text": "This is a small local test content for RAG-Anything with Ollama integration.",
             "page_idx": 0}
        ]
        await self.rag.insert_content_list(
            content_list=content_list,
            file_path="sample.txt",
            doc_id=f"sample-{uuid.uuid4()}",
            display_stats=True
        )

    async def query_loop(self):
        """Simple interactive query loop; uses aquery (async) for multimodal/hybrid queries."""
        if not self.rag:
            raise RuntimeError("RAG not initialized")
        print("\n--- Enter queries (type exit to quit). You can request VLM analysis by mentioning 'analyze image' or similar ---")
        while True:
            q = input("\nYou: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break

            # Choose hybrid mode by default (retrieval + generation)
            mode = "hybrid"
            try:
                # If user asks for VLM-specific analysis, we keep vision enhancement enabled
                res = await self.rag.aquery(q, mode=mode)
                print("\nAssistant:", res)
            except Exception as e:
                print("Query failed:", e)

# ---- Main run entrypoint ----
async def main():
    print("Starting Ollama + RAG-Anything multimodal pipeline")
    print("Text model:", TEXT_MODEL, "| Vision model:", VISION_MODEL, "| Embedding:", EMBEDDING_MODEL)
    integr = OllamaRAGMultimodal(working_dir=WORKING_DIR, docs_dir=DOCS_DIR, max_concurrent=MAX_CONCURRENT)

    if not await integr.test_ollama():
        print("Ollama test failed. Make sure `ollama serve` is running and models are pulled.")
        return

    if not await integr.initialize():
        print("Failed to initialize RAG; aborting.")
        return

    # Insert a small sample so we can query immediately
    await integr.insert_sample_content()

    # Process all files in docs folder (if any)
    docs_path = Path(DOCS_DIR)
    if docs_path.exists() and any(docs_path.iterdir()):
        print("Ingesting docs from", docs_path)
        await integr.ingest_folder(docs_path)
    else:
        print("No docs to ingest in", docs_path, " — create ./docs and drop files (pdf/docx/png/txt)")

    # Enter interactive query loop
    await integr.query_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted by user")
