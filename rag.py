# rag_ollama_multimodal.py
"""
Full multimodal RAG-Anything + Ollama integration (single-file).
Fixed + Stable version.
"""

import os
import sys
import asyncio
import base64
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# ---- Environment defaults ----
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
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
except Exception as e:
    print("Missing required libraries. Install with:\n"
          "pip install --no-cache-dir lightrag==0.4.2 raganything==0.1.31\n")
    raise


# --------------------------------------------------------------------------
#  -------- FIXED WORKING EMBEDDING WRAPPER (CRITICAL FIX) -----------------
# --------------------------------------------------------------------------

async def ollama_embed_async(texts: List[str]) -> List[List[float]]:
    resp = await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE,
        api_key="ollama",
    )

    # Case 1: numpy-like array
    if hasattr(resp, "tolist"):
        return resp.tolist()

    # Case 2: dict (OpenAI embedding format)
    if isinstance(resp, dict) and "data" in resp:
        return [item["embedding"] for item in resp["data"]]

    # Case 3: already list of vectors
    if isinstance(resp, list):
        return resp

    raise ValueError(f"Unknown embedding format from Ollama: {resp}")


def embedding_func_factory():
    return EmbeddingFunc(
        embedding_dim=1024,  # <-- FIXED (nomic-embed-text latest = 1024)
        max_token_size=8192,
        func=ollama_embed_async,
    )


# --------------------------------------------------------------------------
#  -------- LLM / VLM WRAPPERS FOR OLLAMA ---------------------------------
# --------------------------------------------------------------------------

async def ollama_text_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs) -> str:

    return await openai_complete_if_cache(
        model=TEXT_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        temperature=0.0,
        base_url=OLLAMA_BASE,
        api_key="ollama",
        **kwargs,
    )


async def ollama_vision_complete(
        prompt: str,
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

    multimodal = [
        {"role": "system", "content": system_prompt or "You are a helpful vision model."},
        {"role": "user",
         "content": [
             {"type": "text", "text": prompt},
             {"type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
              }
         ]}
    ]

    return await openai_complete_if_cache(
        model=VISION_MODEL,
        prompt="",
        system_prompt=None,
        history_messages=[],
        messages=multimodal,
        base_url=OLLAMA_BASE,
        api_key="ollama",
        **kwargs,
    )


# --------------------------------------------------------------------------
#  -------- MAIN MULTIMODAL RAG CLASS -------------------------------------
# --------------------------------------------------------------------------

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
                    data = await r.json()
                    print("Ollama models:", [m.get("name") for m in data.get("data", [])])
            return True
        except Exception as e:
            print("Ollama unreachable:", e)
            return False

    async def initialize(self) -> bool:
        try:
            print("Initializing RAG-Anything with Ollama...")
            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=ollama_text_complete,
                vision_model_func=ollama_vision_complete,
                embedding_func=embedding_func_factory(),
            )
            async def _noop(*args, **kwargs): return None
            self.rag._mark_multimodal_processing_complete = _noop
            print("RAG initialized.")
            return True
        except Exception as e:
            print("RAG init failed:", e)
            return False

    async def process_single_file(self, file_path: Path):
        try:
            print(f"Processing {file_path} ...")
            await self.rag.process_document_complete(
                file_path=str(file_path),
                output_dir=os.path.join(self.config.working_dir, "output"),
                parse_method="auto",
                display_stats=True,
            )
            print(f"Done: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    async def ingest_folder(self, folder: Path):
        files = [p for p in folder.rglob("*")
                 if p.suffix.lower() in {
                     ".pdf", ".docx", ".doc", ".pptx", ".ppt",
                     ".xlsx", ".xls", ".txt", ".md",
                     ".png", ".jpg", ".jpeg"
                 }]

        if not files:
            print("No supported files.")
            return

        print(f"Found {len(files)} files. Processing...")

        sem = asyncio.Semaphore(self.max_concurrent)

        async def worker(f):
            async with sem:
                await self.process_single_file(f)

        await asyncio.gather(*(worker(f) for f in files))

    async def insert_sample_content(self):
        content_list = [
            {"type": "text",
             "text": "Sample content for local testing.",
             "page_idx": 0}
        ]
        await self.rag.insert_content_list(
            content_list=content_list,
            file_path="sample.txt",
            doc_id=f"sample-{uuid.uuid4()}",
            display_stats=True
        )

    async def query_loop(self):
        print("\n--- Enter query (exit to stop) ---")
        while True:
            q = input("\nYou: ").strip()
            if q.lower() == "exit":
                break
            try:
                res = await self.rag.aquery(q, mode="hybrid")
                print("\nAssistant:", res)
            except Exception as e:
                print("Query error:", e)


# --------------------------------------------------------------------------
#  -------- MAIN ENTRY -----------------------------------------------------
# --------------------------------------------------------------------------

async def main():
    print("Starting Multimodal RAG + Ollama")
    integr = OllamaRAGMultimodal()

    if not await integr.test_ollama():
        return

    if not await integr.initialize():
        return

    await integr.insert_sample_content()

    docs = Path(DOCS_DIR)
    if docs.exists() and any(docs.iterdir()):
        print("Ingesting docs from folder:", docs)
        await integr.ingest_folder(docs)
    else:
        print("Docs folder empty.")

    await integr.query_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
