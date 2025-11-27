# rag_ollama_multimodal.py
"""
BEST MULTIMODAL RAG PIPELINE:
 - Embeddings: ColNomic/ColNomic-embed-multimodal-7B (HuggingFace)
 - LLM + Vision: LLaVA-34B (Ollama)
 - Parser: RAG-Anything (MinerU)
 - Retrieval: FAISS (inner-product)
"""

import os
import asyncio
import uuid
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
import faiss
from transformers import AutoProcessor, AutoModel

# RAG-Anything + LightRAG
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_BASE = "http://localhost:11434/v1"
LLM_MODEL = "llava:34b"          # used for BOTH text + vision

HF_EMBED_MODEL = "ColNomic/ColNomic-embed-multimodal-7B"  # BEST EMBEDDING MODEL

DOCS_DIR = "./docs"
WORK_DIR = "./rag_storage_ollama"


# -----------------------------
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
