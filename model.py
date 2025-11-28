
import urllib.request
import json
import re
import asyncio
import logging
import httpx
logger = logging.getLogger(__name__)

class LLMHelper:
    def __init__(self, model, maxtokens, temperature, url):
        self.model = model
        self.maxtokens = maxtokens
        self.temperature = temperature
        self.url = url

    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        stream: bool = False
    ) -> str | dict:
        """
        Sends request to local LLM server and returns generated response.
        Supports optional streaming.
        """
        if stream:
            return self._stream_response(user_prompt, system_prompt)

        try:
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "options": {
                    "seed": 478,
                    "temperature": self.temperature,
                    "num_ctx": self.maxtokens
                }
            }

            payload = json.dumps(data).encode("utf-8")
            request = urllib.request.Request(self.url, data=payload, method="POST")
            request.add_header("Content-Type", "application/json")

            response_data = ""
            with urllib.request.urlopen(request) as response:
                while True:
                    line = response.readline().decode("utf-8")
                    if not line:
                        break
                    response_json = json.loads(line)
                    response_data += response_json["message"]["content"]

            cleaned = re.sub(r"<think>.*?</think>", "", response_data, flags=re.DOTALL).strip()
            return cleaned

        except Exception as e:
            return {"status": "error", "message": f"Error generating response: {str(e)}"}
