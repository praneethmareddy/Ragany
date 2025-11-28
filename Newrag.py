"""
Final multimodal RAGAnything + Ollama (LLaVA-34B)
 - Uses local Ollama /api/chat for text+vision (llava:34b)
 - Uses local Ollama /api/embeddings for embeddings (nomic-embed-text -> 768 dims)
 - Uses RAGAnything for parsing and indexing
 - Renders PDF pages to images (PyMuPDF) and inserts visual chunks
 - Insert image captions (via LLaVA) into the index for better retrieval

Adjust environment variables below or export them before running.
"""

import os
import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict
from io import BytesIO

import httpx
import fitz  # PyMuPDF
from PIL import Image

# RAGAnything / LightRAG imports
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.utils import EmbeddingFunc
except Exception as e:
    print("Missing required packages. Install with:")
    print("pip install raganything==0.1.31 lightrag==0.4.2 python-dotenv httpx pymupdf pillow aiofiles")
    raise

# -------------------------
# Environment / Defaults
# -------------------------
os.environ.setdefault("LLM_BINDING_HOST", "http://localhost:11434")
os.environ.setdefault("TEXT_VISION_MODEL", "llava:34b")
os.environ.setdefault("EMBED_MODEL", "nomic-embed-text:latest")  # nomic via Ollama -> 768 dims
os.environ.setdefault("WORKING_DIR", "./rag_storage")
os.environ.setdefault("DOCS_DIR", "./docs")
os.environ.setdefault("MAX_CONCURRENT_FILES", "3")
os.environ.setdefault("LLAVA_PAGE_CAPTION", "true")  # if "false", skip immediate LLaVA captioning (faster)

OLLAMA_BASE = os.getenv("LLM_BINDING_HOST").rstrip("/")
TEXT_VISION_MODEL = os.getenv("TEXT_VISION_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
WORKING_DIR = os.getenv("WORKING_DIR")
DOCS_DIR = os.getenv("DOCS_DIR")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_FILES", "3"))
DO_PAGE_CAPTION = os.getenv("LLAVA_PAGE_CAPTION", "true").lower() in ("1", "true", "yes")

# -------------------------
# Utilities
# -------------------------
def uuid4_short() -> str:
    """Short uuid for doc ids"""
    return uuid.uuid4().hex[:12]


def pil_image_to_jpeg_b64(img: Image.Image, quality: int = 85) -> str:
    """Convert PIL Image to base64 JPEG string (no data URI prefix)."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------
# LLaVA / Ollama chat helper (supports images)
# -------------------------
class LLMHelper:
    def __init__(self, model: str = TEXT_VISION_MODEL, base_url: str = OLLAMA_BASE, temperature: float = 0.0, num_ctx: int = 4096):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.num_ctx = int(num_ctx)
        self.chat_url = f"{self.base_url}/api/chat"

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, images_base64: Optional[List[str]] = None, timeout: Optional[float] = None) -> str:
        """
        Send a chat request to Ollama. Images if provided should be base64 JPEG strings (without data: prefix).
        This uses streaming to be robust with larger responses, but falls back to full-body parse.
        """
        payload = {
            "model": self.model,
            "messages": [],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx
            }
        }

        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})

        # Build user message; if images present attach "images" field
        user_msg = {"role": "user", "content": prompt}
        if images_base64:
            # Ollama accepts images as data URIs in `images` field
            user_msg["images"] = [f"data:image/jpeg;base64,{b}" for b in images_base64]

        payload["messages"].append(user_msg)

        # Use streaming request via httpx
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", self.chat_url, json=payload, timeout=timeout) as resp:
                    resp.raise_for_status()
                    text_out = ""
                    async for raw_line in resp.aiter_lines():
                        if not raw_line:
                            continue
                        # Ollama streaming often emits JSON lines. Try parse.
                        try:
                            j = json.loads(raw_line)
                        except json.JSONDecodeError:
                            # Append raw if not JSON
                            text_out += raw_line
                            continue

                        # Try common patterns for chat stream chunks
                        # Possible shapes: {"message": {"content": "..."}} or {"choices": [{"message": {"content": "..."}}]}
                        content_piece = ""
                        if isinstance(j, dict):
                            if "message" in j and isinstance(j["message"], dict):
                                # message.content could be string or dict
                                m = j["message"].get("content")
                                if isinstance(m, str):
                                    content_piece = m
                                elif isinstance(m, dict):
                                    # nested content
                                    content_piece = m.get("text") or m.get("content") or ""
                            elif "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                                ch = j["choices"][0]
                                if isinstance(ch, dict):
                                    cm = ch.get("message") or ch.get("delta")
                                    if isinstance(cm, dict):
                                        content_piece = cm.get("content") or cm.get("text") or ""
                            else:
                                # fallback: try to stringify
                                content_piece = j.get("content") or j.get("text") or ""
                        if content_piece:
                            text_out += content_piece
                    return text_out.strip()
            except Exception as exc:
                # Last-resort: try non-streaming full response
                try:
                    r = await client.post(self.chat_url, json=payload, timeout=60)
                    r.raise_for_status()
                    jr = r.json()
                    # Try common shapes
                    if isinstance(jr, dict):
                        if "message" in jr and isinstance(jr["message"], dict):
                            cont = jr["message"].get("content")
                            return cont if isinstance(cont, str) else json.dumps(cont)
                        if "choices" in jr and isinstance(jr["choices"], list) and jr["choices"]:
                            ch = jr["choices"][0]
                            cont = ch.get("message", {}).get("content") or ch.get("text")
                            return cont if isinstance(cont, str) else json.dumps(cont)
                    return str(jr)
                except Exception as exc2:
                    raise RuntimeError(f"Ollama chat failed (stream err={exc}, fallback err={exc2})")


# -------------------------
# Embeddings: batch via Ollama /api/embeddings
# -------------------------
async def ollama_embeddings_batch(texts: List[str], model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE) -> List[List[float]]:
    """
    Calls Ollama embeddings endpoint in batch.
    Expected payloads vary across Ollama versions; this handles common shapes.
    """
    url = f"{base_url}/api/embeddings"
    payload = {"model": model, "input": texts}

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        resp = r.json()

    # Parse common response shapes:
    # - {"data":[{"embedding":[...]} , ...]}
    # - {"embedding": [...]} (single)
    # - [{"embedding":[...]} , ...]
    if isinstance(resp, dict):
        if "data" in resp and isinstance(resp["data"], list):
            return [item.get("embedding") for item in resp["data"]]
        if "embedding" in resp:
            return [resp["embedding"]]
    if isinstance(resp, list):
        # list of embeddings or list of objects with embedding
        if all(isinstance(x, dict) and "embedding" in x for x in resp):
            return [x["embedding"] for x in resp]
        # maybe raw vectors list
        return resp
    raise ValueError(f"Unknown embedding response shape: {resp}")


def embedding_func_factory() -> EmbeddingFunc:
    # nomic-embed-text via Ollama uses 768 dims
    return EmbeddingFunc(embedding_dim=768, max_token_size=8192, func=ollama_embeddings_batch)


# -------------------------
# PDF page rendering to images
# -------------------------
def render_pdf_pages_to_images_b64(pdf_path: str, zoom: float = 2.0, max_pages: Optional[int] = None) -> List[str]:
    """
    Render PDF pages using PyMuPDF and return list of base64-encoded JPEG strings (no data: prefix).
    zoom: scaling factor for page rendering (2.0 gives higher-res)
    max_pages: optionally limit pages rendered
    """
    img_b64_list: List[str] = []
    doc = fitz.open(pdf_path)
    try:
        pages = range(len(doc))
        if max_pages is not None:
            pages = range(min(len(doc), max_pages))
        for i in pages:
            page = doc.load_page(i)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("jpeg")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            img_b64_list.append(img_b64)
    finally:
        doc.close()
    return img_b64_list


# -------------------------
# The main RAG wrapper class
# -------------------------
class OllamaRAGMultimodal:
    def __init__(self, docs_dir: str = DOCS_DIR, working_dir: str = WORKING_DIR, do_page_caption: bool = DO_PAGE_CAPTION):
        self.docs_dir = Path(docs_dir)
        self.working_dir = str(working_dir)
        self.do_page_caption = do_page_caption

        # RAG config
        self.config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        self.rag: Optional[RAGAnything] = None
        self.llm = LLMHelper(model=TEXT_VISION_MODEL, base_url=OLLAMA_BASE, temperature=0.0, num_ctx=4096)

    async def initialize(self):
        print("[RAG] Initializing RAGAnything with Ollama adapters...")
        self.rag = RAGAnything(
            config=self.config,
            llm_model_func=self._text_adapter,
            vision_model_func=self._vision_adapter,
            embedding_func=embedding_func_factory()
        )
        # no-op hook to avoid duplicate multimodal marking in some RAGAnything versions
        async def _noop(*a, **k): return None
        self.rag._mark_multimodal_processing_complete = _noop
        print("[RAG] Initialized.")

    # adapter for text-only calls
    async def _text_adapter(self, prompt: str, system_prompt: Optional[str] = None, history_messages: Optional[List[Dict]] = None, **kwargs):
        return await self.llm.generate_response(prompt=prompt, system_prompt=(system_prompt or ""))

    # adapter for vision calls (RAGAnything will use this to ask about images)
    async def _vision_adapter(self, prompt: str, system_prompt: Optional[str] = None, image_base64: Optional[str] = None, messages: Optional[List[Dict]] = None, **kwargs):
        # If a single image_base64 provided, forward it; if messages provided, fall back to constructing request similarly
        if image_base64:
            return await self.llm.generate_response(prompt=prompt, system_prompt=(system_prompt or ""), images_base64=[image_base64])
        # messages fallback (pass-through not implemented in detail here)
        return await self.llm.generate_response(prompt=prompt, system_prompt=(system_prompt or ""))

    async def process_single_file(self, file_path: Path):
        print(f"[INGEST] Processing: {file_path}")
        try:
            # 1) Let RAGAnything parse text/tables/slides via mineru
            await self.rag.process_document_complete(
                file_path=str(file_path),
                output_dir=os.path.join(self.working_dir, "output"),
                parse_method="auto",
                display_stats=True
            )

            # 2) For PDFs: render pages -> insert image chunks + optional captions
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                print(f"[INGEST] Rendering PDF pages for: {file_path.name}")
                try:
                    images_b64 = render_pdf_pages_to_images_b64(str(file_path), zoom=1.5)
                except Exception as e:
                    print(f"[ERROR] PDF rendering failed for {file_path}: {e}")
                    images_b64 = []

                if images_b64:
                    content_list = []
                    for idx, img_b64 in enumerate(images_b64):
                        # Add image chunk
                        content_list.append({
                            "type": "image",
                            "text": f"{file_path.name} - page {idx+1}",
                            "image_base64": img_b64,
                            "page_idx": idx
                        })

                        # Optionally caption page using LLaVA to improve searchability
                        if self.do_page_caption:
                            try:
                                caption = await self._vision_adapter(
                                    prompt="Provide a short caption (1-2 lines) describing the main content of the page and list any charts/tables present.",
                                    system_prompt="You are a helpful document vision assistant. Keep captions concise.",
                                    image_base64=img_b64
                                )
                            except Exception as e:
                                caption = f"[captioning failed: {e}]"
                            content_list.append({
                                "type": "text",
                                "text": caption or f"Page {idx+1} of {file_path.name}",
                                "page_idx": idx
                            })

                    if content_list:
                        doc_id = f"{file_path.stem}-pages-{uuid4_short()}"
                        await self.rag.insert_content_list(
                            content_list=content_list,
                            file_path=str(file_path),
                            doc_id=doc_id,
                            display_stats=True
                        )
                        print(f"[INGEST] Inserted {len(content_list)} visual/text chunks for {file_path.name}")

            # 3) For image files (jpg/png etc.) -> insert and caption
            elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".webp"}:
                with open(file_path, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("utf-8")
                caption = None
                if self.do_page_caption:
                    try:
                        caption = await self._vision_adapter(
                            prompt="Provide a short caption and list any notable items (tables, charts, text) in this image.",
                            system_prompt="You are a helpful vision assistant. Be concise.",
                            image_base64=b64
                        )
                    except Exception as e:
                        caption = f"[caption error: {e}]"

                content_list = [
                    {"type": "image", "text": file_path.name, "image_base64": b64},
                    {"type": "text", "text": caption or f"Image: {file_path.name}"}
                ]
                await self.rag.insert_content_list(content_list=content_list, file_path=str(file_path), doc_id=f"{file_path.stem}-img-{uuid4_short()}")
                print(f"[INGEST] Inserted image chunks for {file_path.name}")

            # 4) For other types (docx, pptx, xlsx, txt, md) the mineru parse covered them.
        except Exception as e:
            print(f"[ERROR] ingesting {file_path}: {e}")

    async def process_folder(self, folder: Optional[str] = None):
        folder = Path(folder or self.docs_dir)
        supported_suffixes = {
            ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
            ".png", ".jpg", ".jpeg", ".tiff", ".webp", ".txt", ".md"
        }
        files = [p for p in folder.rglob("*") if p.suffix.lower() in supported_suffixes]

        if not files:
            print("[INGEST] No supported files found.")
            return

        print(f"[INGEST] Found {len(files)} files. Ingesting with concurrency={MAX_CONCURRENT} ...")
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def worker(p: Path):
            async with sem:
                await self.process_single_file(p)

        await asyncio.gather(*(worker(f) for f in files))
        print("[INGEST] Folder ingestion complete.")

    async def query(self, q: str, mode: str = "hybrid"):
        print(f"[QUERY] {q}")
        res = await self.rag.aquery(q, mode=mode)
        return res


# -------------------------
# Main
# -------------------------
async def main():
    ragm = OllamaRAGMultimodal(docs_dir=DOCS_DIR, working_dir=WORKING_DIR, do_page_caption=DO_PAGE_CAPTION)
    await ragm.initialize()

    # Ingest all files in the docs folder (change DOCS_DIR env var or pass path)
    await ragm.process_folder(DOCS_DIR)

    # Example queries
    q1 = "Summarize quarterly financial highlights across all reports."
    try:
        ans1 = await ragm.query(q1)
        print("\n----- ANSWER 1 -----\n", ans1)
    except Exception as e:
        print("Query 1 error:", e)

    q2 = "List pages or images that contain charts or tables and summarize them."
    try:
        ans2 = await ragm.query(q2)
        print("\n----- ANSWER 2 -----\n", ans2)
    except Exception as e:
        print("Query 2 error:", e)


if __name__ == "__main__":
    asyncio.run(main())
