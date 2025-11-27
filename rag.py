# rag_ollama_multimodal.py
"""
Full multimodal RAG-Anything + Ollama integration (single-file).
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
os.environ.setdefault("LLM_BINDING_HOST", "http://localhost:11434")
os.environ.setdefault("LLM_MODEL", "gpt-oss:20b")
os.environ.setdefault("VISION_MODEL", "llava:34b")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text:latest")
os.environ.setdefault("WORKING_DIR", "./rag_storage_ollama")
os.environ.setdefault("DOCS_DIR", "./docs")
os.environ.setdefault("MAX_CONCURRENT_FILES", "3")

load_dotenv()

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
    from lightrag.utils import EmbeddingFunc, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
except Exception as e:
    print("Missing required libraries. Install with:\n"
          "pip install --no-cache-dir lightrag==0.4.2 raganything==0.1.31\n")
    raise

# ---- Helper: Ollama OpenAI-compatible wrappers ----
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

# ---- Embedding function ----
def embedding_func_factory():
    return EmbeddingFunc(
        embedding_dim=768,
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
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        self.docs_dir = Path(docs_dir)
        self.rag: Optional[RAGAnything] = None
        self.max_concurrent = max_concurrent

    async def test_ollama(self) -> bool:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"{OLLAMA_BASE}/models") as r:
                    if r.status != 200:
                        return False
                    return True
        except Exception:
            return False

    async def initialize(self) -> bool:
        try:
            print("Initializing RAG-Anything...")
            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=ollama_text_complete,
                vision_model_func=ollama_vision_complete,
                embedding_func=embedding_func_factory(),
            )
            async def _noop(*args, **kwargs): return None
            self.rag._mark_multimodal_processing_complete = _noop
            return True
        except Exception as e:
            print("Failed to initialize:", e)
            return False

    async def process_single_file(self, file_path: Path):
        if not self.rag:
            raise RuntimeError("RAG not initialized")
        print(f"Processing {file_path} ...")
        await self.rag.process_document_complete(
            file_path=str(file_path),
            output_dir=os.path.join(self.config.working_dir, "output"),
            parse_method="auto",
            display_stats=True,
        )
        print(f"Completed {file_path}")

    async def ingest_folder(self, folder: Path):
        files = [p for p in folder.rglob("*") if p.suffix.lower() in
                 {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md", ".png", ".jpg", ".jpeg"}]

        if not files:
            print("No supported files found.")
            return

        sem = asyncio.Semaphore(self.max_concurrent)

        async def worker(p):
            async with sem:
                await self.process_single_file(p)

        print(f"Found {len(files)} files. Ingesting...")
        await asyncio.gather(*(worker(p) for p in files))

    async def insert_sample_content(self):
        if not self.rag:
            raise RuntimeError("RAG not initialized")

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

    # -------------------------------------------------------------
    # 🔥 UPDATED QUERY LOOP — REAL RAG (Retrieval + Answering)
    # -------------------------------------------------------------
    async def query_loop(self):
        if not self.rag:
            raise RuntimeError("RAG not initialized")

        print("\n--- RAG Ready. Type 'exit' to quit. ---")

        while True:
            q = input("\nYou: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break

            try:
                # STEP 1 — Retrieve context
                print("\n[1/2] Retrieving relevant chunks...")
                ctx = await self.rag.aquery(
                    q,
                    param=QueryParam(mode="hybrid", only_need_context=True)
                )

                print("\n🔍 Retrieved Context (Truncated):")
                print(str(ctx)[:1500])
                print("\n" + "-" * 50)

                # STEP 2 — Generate final answer
                print("\n[2/2] Generating answer...")
                ans = await self.rag.aquery(
                    q,
                    param=QueryParam(mode="hybrid")
                )

                print("\n🤖 Assistant:", ans)

            except Exception as e:
                print("Query failed:", e)

# ---- Main runner ----
async def main():
    integr = OllamaRAGMultimodal(working_dir=WORKING_DIR, docs_dir=DOCS_DIR)

    if not await integr.test_ollama():
        print("Ollama check failed.")
        return

    if not await integr.initialize():
        return

    await integr.insert_sample_content()

    docs_path = Path(DOCS_DIR)
    if docs_path.exists() and any(docs_path.iterdir()):
        await integr.ingest_folder(docs_path)
    else:
        print("No docs found in ./docs")

    await integr.query_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted.")
