# main_rag.py
import asyncio
import json
import httpx
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_embed

from model import LLMHelper

PDF_PATH = "./docs/Kubernetes.pdf"


# -------- TEXT LLM (GPT-OSS-20B) --------
async def generate_response_v2(
    prompt: str,
    system_prompt=None,
    history_messages=None,
    model="gpt-oss:20b",
    host="http://localhost:11434/api/chat"
):
    history_messages = history_messages or []

    llm = LLMHelper(
        model=model,
        maxtokens=32768,
        temperature=0.2,
        url=host,
        timeout=120
    )

    return await llm.generate_response(
        user_prompt=prompt,
        system_prompt=system_prompt,
        extra_messages=history_messages
    )


# -------- VISION LLM (LLAVA-34B) --------
async def vision_model(prompt: str, image_path: str = None):
    """
    Vision model (LLAVA 34B) for image processors.
    """
    payload = {
        "model": "llava:34b",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                **({"images": [str(image_path)]} if image_path else {})
            }
        ]
    }

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post("http://localhost:11434/api/chat", json=payload)
        r.raise_for_status()

        try:
            j = r.json()
            if "message" in j and "content" in j["message"]:
                return j["message"]["content"]
            if "choices" in j and j["choices"]:
                return j["choices"][0].get("message", {}).get("content", "")
            return json.dumps(j)
        except:
            return r.text


# -------- EMBEDDING FUNCTION (GPU nomic-embed-text) --------
embedding_func = EmbeddingFunc(
    embedding_dim=768,
    max_token_size=8192,
    func=lambda texts: ollama_embed(
        texts,
        embed_model="nomic-embed-text:latest",
        host="http://localhost:11434"
    ),
)


# ------------------ MAIN ------------------
async def main():
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",

        # multimodal ON
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,

        # GPU acceleration
        ocr_use_gpu=True,
        use_gpu=True,

        # speed settings
        embedding_batch_size=32,
        embedding_workers=8,
        max_concurrent_files=4,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=generate_response_v2,   # GPT-OSS-20B
        vision_model_func=vision_model,       # LLAVA-34B
        embedding_func=embedding_func,
    )

    # Initialize LightRAG + storages
    init_res = await rag._ensure_lightrag_initialized()
    if not init_res.get("success", False):
        print("Initialization error:", init_res)
        return

    # Fast ingest (chunked mode)
    await rag.process_document(
        file_path=PDF_PATH,
        parse_method="auto"
    )

    # Hybrid retrieval + GPT-OSS-20B answer
    result = await rag.aquery(
        "What are SOLR’s weaknesses?",
        mode="hybrid"
    )
    print("\nANSWER:\n", result)


if __name__ == "__main__":
    asyncio.run(main())        working_dir="./rag_storage",
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=False
       # enable_rerank=False
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=generate_response_v2,
        embedding_func=embedding_func
       # rerank_model_func=rerank_model_func
    )

    # Process the PDF document
    await rag.process_document_complete(
        file_path=pdf_path,
        output_dir="./output",
        parse_method="auto"
    )

    # Text query
    text_result = await rag.aquery(
        "what are SOLR’S WEAKNESSES?",
        mode="hybrid"
    )
    print("Text query result:\n", text_result)
 
if __name__ == "__main__":
    asyncio.run(main())
