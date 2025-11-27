# rag_ollama_multimodal.py
"""
Full multimodal RAG-Anything + Ollama integration (A6000 Optimized).
Requirements:
 - Ollama running: `ollama serve`
 - Models: qwen2.5:32b, llama3.2-vision, mxbai-embed-large
 - Pip: raganything==0.1.31, lightrag==0.4.2
"""

import os
import sys
import asyncio
import base64
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# ---- Environment defaults (Optimized for RTX A6000 / 48GB VRAM) ----
os.environ.setdefault("LLM_BINDING", "ollama")
os.environ.setdefault("LLM_BINDING_HOST", "http://localhost:11434")
os.environ.setdefault("LLM_MODEL", "qwen2.5:32b")             # 32B Param Model (Logic/Text)
os.environ.setdefault("VISION_MODEL", "llama3.2-vision")       # 11B Vision Model (Charts/Images)
os.environ.setdefault("EMBEDDING_MODEL", "mxbai-embed-large")  # High-fidelity embeddings
os.environ.setdefault("WORKING_DIR", "./rag_storage_ollama")
os.environ.setdefault("DOCS_DIR", "./docs")
os.environ.setdefault("MAX_CONCURRENT_FILES", "1") # Keep 1 for safety, we loop manually anyway

load_dotenv()

OLLAMA_BASE = os.getenv("LLM_BINDING_HOST").rstrip("/") + "/v1"
TEXT_MODEL = os.getenv("LLM_MODEL")
VISION_MODEL = os.getenv("VISION_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
WORKING_DIR = os.getenv("WORKING_DIR")
DOCS_DIR = os.getenv("DOCS_DIR")

# ---- Imports from RAG-Anything / LightRAG ----
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.utils import EmbeddingFunc, QueryParam  # Added QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
except Exception as e:
    print("Missing required libraries.\nRun: pip install lightrag==0.4.2 raganything==0.1.31 magic-pdf[full]")
    raise

# ---- Helper: Ollama OpenAI-compatible wrappers (async) ----
async def ollama_text_complete(prompt: str,
                               system_prompt: Optional[str] = None,
                               history_messages: Optional[List[Dict]] = None,
                               **kwargs) -> str:
    return await openai_complete_if_cache(
        model=TEXT_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=OLLAMA_BASE,
        api_key="ollama",
        **kwargs,
    )

async def ollama_embed_async(texts: List[str]) -> List[List[float]]:
    embeds = await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE,
        api_key="ollama",
    )
    try:
        return embeds.tolist()
    except Exception:
        return embeds

# ---- Vision model wrapper ----
async def ollama_vision_complete(prompt: str,
                                 system_prompt: Optional[str] = None,
                                 history_messages: Optional[List[Dict]] = None,
                                 image_base64: Optional[str] = None,
                                 messages: Optional[List[Dict]] = None,
                                 **kwargs) -> str:
    if messages:
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

    # Manual message construction for single image analysis
    multimodal_msg = [
        {"role": "system", "content": system_prompt or "You are a helpful vision analyst. Describe this image in detail."},
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

def embedding_func_factory():
    return EmbeddingFunc(
        embedding_dim=1024, # mxbai-embed-large uses 1024 dims (unlike nomic's 768)
        max_token_size=8192,
        func=ollama_embed_async,
    )

# ---- Main Integration Class ----
class OllamaRAGMultimodal:
    def __init__(self,
                 working_dir: str = WORKING_DIR,
                 docs_dir: str = DOCS_DIR):
        self.config = RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru", 
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        self.docs_dir = Path(docs_dir)
        self.rag: Optional[RAGAnything] = None

    async def test_ollama(self) -> bool:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"{OLLAMA_BASE}/models") as r:
                    if r.status != 200:
                        return False
                    data = await r.json()
                    available = [m.get("id") or m.get("name") for m in data.get("data", [])]
                    print(f"Checking models... Text: {TEXT_MODEL} | Vision: {VISION_MODEL}")
                    return True
        except Exception as e:
            print("Failed to contact Ollama:", e)
            return False

    async def initialize(self) -> bool:
        try:
            print("Initializing RAG-Anything (A6000 Config)...")
            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=ollama_text_complete,
                vision_model_func=ollama_vision_complete,
                embedding_func=embedding_func_factory(),
            )
            # Patch for LightRAG compatibility
            async def _noop_mark(*args, **kwargs): return None
            self.rag._mark_multimodal_processing_complete = _noop_mark
            return True
        except Exception as e:
            print("Failed to initialize RAGAnything:", e)
            return False

    async def process_single_file(self, file_path: Path):
        if not self.rag: raise RuntimeError("RAG not initialized")
        print(f"Processing {file_path.name}...")
        await self.rag.process_document_complete(
            file_path=str(file_path),
            output_dir=os.path.join(self.config.working_dir, "output"),
            parse_method="auto",
            display_stats=True,
        )
        print(f"Completed {file_path.name}")

    async def ingest_folder_sequential(self, folder: Path):
        """
        SEQUENTIAL INGESTION (Fixes the concurrent write bug).
        Safe for Knowledge Graph construction.
        """
        files = [p for p in folder.rglob("*") if p.suffix.lower() in
                 {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".png", ".jpg", ".jpeg"}]
        
        if not files:
            print("No supported files found in", folder)
            return

        print(f"Found {len(files)} files. Processing sequentially...")
        for i, p in enumerate(files):
            try:
                print(f"--- File {i+1}/{len(files)} ---")
                await self.process_single_file(p)
            except Exception as e:
                print(f"Failed to process {p.name}: {e}")
        print("All files ingested.")

    async def insert_sample_content(self):
        if not self.rag: raise RuntimeError("RAG not initialized")
        print("Inserting sample content...")
        content_list = [{"type": "text", "text": "RAG-Anything A6000 Test initialized.", "page_idx": 0}]
        await self.rag.insert_content_list(
            content_list=content_list, file_path="system_init.txt", 
            doc_id=f"sys-{uuid.uuid4()}"
        )

    async def query_loop(self):
        """Interactive loop: Prints References -> Then Answer."""
        if not self.rag: raise RuntimeError("RAG not initialized")
        print("\n--- A6000 RAG Ready. Type 'exit' to quit. ---")
        
        while True:
            q = input("\nYou: ").strip()
            if not q: continue
            if q.lower() in ("exit", "quit"): break

            # Mode: Hybrid is best for Qwen2.5-32B to synthesize graph + vector data
            mode = "hybrid"

            try:
                # PASS 1: Get Context Only
                print("\n[1/2] Retrieving Documents & Images...")
                context_param = QueryParam(mode=mode, only_need_context=True)
                refs = await self.rag.aquery(q, param=context_param)
                
                print(f"\n--- 🔍 RETRIEVED REFERENCES ({mode}) ---")
                if isinstance(refs, str):
                    # Show first 1500 chars to avoid flooding terminal
                    print(refs[:1500] + ("\n...[Truncated]..." if len(refs)>1500 else ""))
                else:
                    print(refs)
                print("-" * 40)

                # PASS 2: Generate Answer
                print("\n[2/2] Generating Answer with Qwen2.5-32B...")
                answer_param = QueryParam(mode=mode)
                res = await self.rag.aquery(q, param=answer_param)
                print(f"\n🤖 Assistant:\n{res}")

            except Exception as e:
                print(f"Error: {e}")

# ---- Run ----
async def main():
    print(f"Pipeline Config: Text={TEXT_MODEL} | Vision={VISION_MODEL} | Embed={EMBEDDING_MODEL}")
    integr = OllamaRAGMultimodal(working_dir=WORKING_DIR, docs_dir=DOCS_DIR)

    if not await integr.test_ollama():
        print("Ollama check failed.")
        return

    if not await integr.initialize():
        return

    # Check if we need to ingest (only if storage dir is empty/new)
    # Note: If you want to force re-ingest, delete 'rag_storage_ollama' folder manually
    docs_path = Path(DOCS_DIR)
    if docs_path.exists() and any(docs_path.iterdir()):
        # We always run ingestion scan. LightRAG skips duplicates if file_hash matches.
        await integr.ingest_folder_sequential(docs_path)
    else:
        # Fallback if no docs
        await integr.insert_sample_content()

    await integr.query_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted.")
