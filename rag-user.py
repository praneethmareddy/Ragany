import asyncio

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_embed
import json
import urllib.request
import re
import httpx
from model import LLMHelper
pdf_path = r"./docs/Kubernetes.pdf"
# excel_path = r"D:\Data\Search.docx"


async def generate_response_v2(
       prompt, system_prompt=None, history_messages=[], model="llama3.1:8b-instruct-q8_0", **kwargs
    ) -> str | dict:
    llm = LLMHelper(model,32768,0.3,"http://localhost:11434/api/chat")
    response_text = await llm.generate_response(prompt, system_prompt)
    return response_text


# Async LLM function for Ollama

# Embedding function
embedding_func = EmbeddingFunc(
    embedding_dim=768,
    max_token_size=8192,
    func=lambda texts: ollama_embed(
        texts, embed_model="nomic-embed-text:latest", host="http://localhost:11434"
    ),
)

async def main():
    # RAG config
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
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
