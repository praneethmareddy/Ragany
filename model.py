# model.py
import json
import re
import logging
from typing import Union, Optional, List
import httpx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMHelper:
    """
    Async LLM helper using httpx.AsyncClient.
    Works with Ollama, vLLM HTTP server, LM Studio API, or any OpenAI-compatible endpoint.
    """

    def __init__(self, model: str, maxtokens: int, temperature: float, url: str, timeout: int = 60):
        self.model = model
        self.maxtokens = maxtokens
        self.temperature = temperature
        self.url = url
        self.timeout = timeout

    async def _parse_response(self, text: str) -> str:
        """
        Parses JSON, OpenAI-style responses, or NDJSON streaming fallback.
        """
        # Try regular JSON
        try:
            j = json.loads(text)
            if isinstance(j, dict):
                if "message" in j and "content" in j["message"]:
                    return j["message"]["content"]
                if "choices" in j and j["choices"]:
                    c = j["choices"][0]
                    return c.get("message", {}).get("content") or c.get("text", "")
            return text
        except json.JSONDecodeError:
            # Try NDJSON streaming fallback
            output = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    if "message" in j and "content" in j["message"]:
                        output.append(j["message"]["content"])
                    elif "text" in j:
                        output.append(j["text"])
                except Exception:
                    output.append(line)
            return "\n".join(output)

    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        extra_messages: Optional[List[dict]] = None,
    ) -> Union[str, dict]:
        """
        Sends request to local LLM server and returns the response text.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            if extra_messages:
                messages.extend(extra_messages)

            payload = {
                "model": self.model,
                "messages": messages,
                "options": {
                    "seed": 42,
                    "temperature": self.temperature,
                    "num_ctx": self.maxtokens,
                },
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(self.url, json=payload)
                r.raise_for_status()
                raw_text = r.text

            parsed = await self._parse_response(raw_text)
            cleaned = re.sub(r"<think>.*?</think>", "", parsed, flags=re.DOTALL).strip()
            return cleaned

        except Exception as e:
            logger.exception("LLM generation error")
            return {"status": "error", "message": str(e)}
