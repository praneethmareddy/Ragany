"""
FINAL MULTIMODAL RAG PIPELINE (Stable + Compatible)
- raganything==1.2.8
- SigLIP SO400M embeddings (public)
- LLaVA-34B via Ollama for responses
"""

import os
import asyncio
import torch
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict

# HuggingFace embedding model
from transformers import AutoProcessor, AutoModel

# RAGAnything v1.2.8
from raganything import RAGAnything, RAGAnythingConfig

# Ollama LLaVA wrapper (OpenAI-compatible)
from lightrag.llm.openai import openai_complete_if_cache


# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
HF_EMBED_MODEL = "google/siglip-so400m-patch14-384"  # BEST public multimodal embedder
OLLAMA_BASE = "http://localhost:11434/v1"
LLM_MODEL = "llava:34b"
DOCS_DIR = "./docs"
WORK_DIR = "./rag_storage_ollama"


print("\n[INIT] Loading SigLIP embeddings:", HF_EMBED_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------
# LOAD SIGLIP MODEL
# -----------------------------------------------------
processor = AutoProcessor.from_pretrained(HF_EMBED_MODEL)
embed_model = AutoModel.from_pretrained(HF_EMBED_MODEL).to(device)
embed_model.eval()

# Determine real embedding dimension
with torch.no_grad():
    t = processor(text="hello", return_tensors="pt").to(device)
    e = embed_model.get_text_features(**t)
    EMBED_DIM = e.shape[-1]

print(f"[INIT] Embedding dimension = {EMBED_DIM}")


# -----------------------------------------------------
# EMBEDDING FUNCTION
# -----------------------------------------------------
def embed_text(texts: List[str]):
    with torch.no_grad():
        batch = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        vecs = embed_model.get_text_features(**batch)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().numpy().astype("float32")


# -----------------------------------------------------
# FAISS INDEX + STORE
# -----------------------------------------------------
index = faiss.IndexFlatIP(EMBED_DIM)
chunk_store: List[Dict] = []


def add_chunks(chunks: List[Dict]):
    for c in chunks:
        emb = embed_text([c["text"]])[0]
        index.add(np.array([emb]))
        chunk_store.append(c)


def search_chunks(query: str, k=6):
    q_emb = embed_text([query])
    scores, idxs = index.search(q_emb, k)

    results = []
    for sc, ix in zip(scores[0], idxs[0]):
        if ix < 0:
            continue
        entry = chunk_store[ix]
        entry["score"] = float(sc)
        results.append(entry)

    return results


# -----------------------------------------------------
# LLaVA CALL (Ollama)
# -----------------------------------------------------
async def llava_answer(prompt: str) -> str:
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        base_url=OLLAMA_BASE,
        api_key="ollama",
        temperature=0.1
    )


# -----------------------------------------------------
# RAGANYTHING PARSER (SAFE FOR v1.2.8)
# -----------------------------------------------------
rag_cfg = RAGAnythingConfig(
    working_dir=WORK_DIR,
    parser="mineru",
    parse_method="auto",
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
)

rag = RAGAnything(config=rag_cfg)


async def ingest_documents():
    print("\n[INFO] Starting document ingestion...")

    valid_ext = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}

    for f in Path(DOCS_DIR).rglob("*"):
        if f.suffix.lower() not in valid_ext:
            continue

        print(f"[INGEST] Parsing: {f}")

        try:
            result = await rag.process_document_complete(
                file_path=str(f),
                parse_method="auto",
                display_stats=False
            )
        except Exception as e:
            print(f"[ERROR] Failed to parse {f}: {e}")
            continue

        # SAFETY: skip if mineru returned nothing
        if result is None or "chunks" not in result:
            print(f"[WARN] No chunks extracted from {f}")
            continue

        chunks = result["chunks"]
        add_chunks(chunks)

    print("\n[INFO] Ingestion complete. Total chunks:", len(chunk_store))


# -----------------------------------------------------
# RAG QUERY PIPELINE
# -----------------------------------------------------
async def answer_query(q: str) -> str:
    retrieved = search_chunks(q, k=6)

    context = ""
    for r in retrieved:
        context += f"[score={r['score']:.4f}] {r['text']}\n"

    prompt = f"""
You are LLaVA-34B, a highly intelligent multimodal reasoner.

Use the following retrieved knowledge:

--- CONTEXT ---
{context}
--- END ---

User query: {q}

Provide the most accurate answer.
"""

    return await llava_answer(prompt)


# -----------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------
async def main():
    print("\n=== SigLIP + LLaVA-34B Multimodal RAG Started ===")

    await ingest_documents()

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        ans = await answer_query(q)
        print("\nAssistant:", ans)


if __name__ == "__main__":
    asyncio.run(main())# -------------------------------------------------------------
print(f"[INFO] Loading embedding model: {HF_EMBED_MODEL}")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(HF_EMBED_MODEL)
embed_model = AutoModel.from_pretrained(HF_EMBED_MODEL).to(device)
embed_model.eval()

# Determine embedding dimension
with torch.no_grad():
    test = processor(text="hello", return_tensors="pt").to(device)
    vec = embed_model.get_text_features(**test)
    EMBED_DIM = vec.shape[-1]

print(f"[INFO] Embedding dimension = {EMBED_DIM}")


# -------------------------------------------------------------
# EMBEDDING FUNCTION
# -------------------------------------------------------------
def embed_text(texts: List[str]):
    with torch.no_grad():
        batch = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        emb = embed_model.get_text_features(**batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
        return emb.cpu().numpy().astype("float32")


# -------------------------------------------------------------
# FAISS INDEX
# -------------------------------------------------------------
index = faiss.IndexFlatIP(EMBED_DIM)  # inner product (cosine)
chunk_store: List[Dict] = []


def add_chunks(chunks: List[Dict]):
    for c in chunks:
        emb = embed_text([c["text"]])[0]
        index.add(np.array([emb]))
        chunk_store.append(c)


def search(query: str, k=5):
    q_emb = embed_text([query])
    scores, idxs = index.search(q_emb, k)
    results = []
    for sc, ix in zip(scores[0], idxs[0]):
        if ix == -1:
            continue
        c = chunk_store[ix]
        c["score"] = float(sc)
        results.append(c)
    return results


# -------------------------------------------------------------
# OLLAMA LLaVA WRAPPER
# -------------------------------------------------------------
async def run_llava(prompt: str):
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        base_url=OLLAMA_BASE,
        api_key="ollama",
        temperature=0.1
    )


# -------------------------------------------------------------
# RAG-ANYTHING PARSING (v1.2.8 compatible)
# -------------------------------------------------------------
rag_cfg = RAGAnythingConfig(
    working_dir=WORK_DIR,
    parser="mineru",
    parse_method="auto",
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
)

rag = RAGAnything(config=rag_cfg)


async def ingest_docs():
    print("[INFO] Ingesting docs...")

    for f in Path(DOCS_DIR).rglob("*"):
        if f.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}:
            print(f"[INFO] Parsing: {f}")

            # v1.2.8 → use process_document_complete
            result = await rag.process_document_complete(
                file_path=str(f),
                parse_method="auto",
                display_stats=False
            )

            chunks = result.get("chunks", [])
            add_chunks(chunks)

    print("[INFO] Document ingestion completed.\n")


# -------------------------------------------------------------
# RAG QUERY
# -------------------------------------------------------------
async def rag_query(q: str):
    retrieved = search(q, k=6)

    ctx = ""
    for r in retrieved:
        ctx += f"\n[score={r['score']:.4f}] {r['text']}\n"

    prompt = f"""
You are LLaVA-34B, a multimodal reasoning model.

Use the following retrieved context:

--- CONTEXT ---
{ctx}
--- END CONTEXT ---

User query: {q}

Give the best answer with maximum accuracy.
"""

    return await run_llava(prompt)


# -------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------
async def main():
    print("=== MULTIMODAL RAG STARTED (SigLIP + LLaVA-34B) ===")

    await ingest_docs()

    while True:
        q = input("\nYou: ")
        if q.lower() in {"exit", "quit"}:
            break
        ans = await rag_query(q)
        print("\nAssistant:", ans)


if __name__ == "__main__":
    asyncio.run(main())# -----------------------------
# LOAD COLNOMIC MULTIMODAL EMBEDDING
# -----------------------------
print(f"\n[INFO] Loading ColNomic Multimodal Embedding Model: {HF_EMBED_MODEL}")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(HF_EMBED_MODEL)
embed_model = AutoModel.from_pretrained(HF_EMBED_MODEL).to(device)
embed_model.eval()

# Detect embedding dimension dynamically
with torch.no_grad():
    sample = processor(text="hello world", return_tensors="pt").to(device)
    vec = embed_model.get_text_features(**sample)
    EMBED_DIM = vec.shape[-1]

print(f"[INFO] Embedding dimension = {EMBED_DIM}\n")


def embed_text(texts: List[str]):
    """Embed text using ColNomic multimodal model."""
    with torch.no_grad():
        batch = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        feats = embed_model.get_text_features(**batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()


# -----------------------------
# FAISS INDEX
# -----------------------------
index = faiss.IndexFlatIP(EMBED_DIM)
chunk_store: List[Dict] = []


def add_to_index(chunks: List[Dict]):
    for ch in chunks:
        emb = embed_text([ch["text"]])[0]
        index.add(np.array([emb], dtype="float32"))
        chunk_store.append(ch)


def search_index(query: str, k: int = 6):
    q_emb = np.array(embed_text([query]), dtype="float32")
    scores, idxs = index.search(q_emb, k)
    results = []
    for sc, ix in zip(scores[0], idxs[0]):
        if ix == -1:
            continue
        hit = chunk_store[ix]
        hit = {**hit, "score": float(sc)}
        results.append(hit)
    return results


# -----------------------------
# OLLAMA LLaVA-34B WRAPPER
# -----------------------------
async def ollama_llava(prompt: str):
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        base_url=OLLAMA_BASE,
        api_key="ollama",
        temperature=0.1
    )


# -----------------------------
# RAG-ANYTHING PARSER
# -----------------------------
rag_cfg = RAGAnythingConfig(
    working_dir=WORK_DIR,
    parser="mineru",
    parse_method="auto",
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
)

rag = RAGAnything(config=rag_cfg)


async def ingest_documents():
    print("[INFO] Ingesting documents...")
    for file in Path(DOCS_DIR).rglob("*"):
        if file.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}:
            print(f"[INFO] Parsing: {file}")
            chunks = await rag.process_document_extract_chunks(str(file))
            add_to_index(chunks)

    print("[INFO] Document ingestion complete.\n")


# -----------------------------
# MULTIMODAL RAG QUERY
# -----------------------------
async def rag_query(q: str):
    retrieved = search_index(q, k=6)

    context = ""
    for r in retrieved:
        context += f"\n[score={r['score']:.4f}] {r['text']}\n"

    prompt = f"""
You are LLaVA-34B, a multimodal expert model.
Use the retrieved context below (from text, images, OCR, tables):

--- CONTEXT START ---
{context}
--- CONTEXT END ---

User query:
{q}

Give a precise, factual, best-quality answer.
"""

    print("\n[INFO] Sending to LLaVA-34B...")
    return await ollama_llava(prompt)


# -----------------------------
# MAIN LOOP
# -----------------------------
async def main():
    print("=== ColNomic + LLaVA-34B Multimodal RAG Started ===")

    await ingest_documents()

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        ans = await rag_query(q)
        print("\nAssistant:", ans)


if __name__ == "__main__":
    asyncio.run(main())
