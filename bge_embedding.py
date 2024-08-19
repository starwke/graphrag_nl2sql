from typing import Dict, List, Optional
from urllib.parse import urljoin

import httpx
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface.utils import format_query, format_text

DEFAULT_URL = "http://127.0.0.1:8080"


class BAAIBGEEmbeddings(BaseEmbedding):
    base_url: str = Field(
        default=DEFAULT_URL,
        description="Base URL for the text embeddings service.",
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the request.",
    )
    subpath: Optional[str] = Field(
        default="/embed",
        description="Subpath of the embedding service API endpoint.",
    )
    customer_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Customer headers to pass to the request.",
    )

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_URL,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        timeout: float = 60.0,
        callback_manager: Optional[CallbackManager] = None,
        subpath: Optional[str] = "/embed",
        customer_headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model_name=model_name, embed_batch_size=embed_batch_size, callback_manager=callback_manager)

        self.base_url = base_url
        self.timeout = timeout

        if not subpath:
            subpath = ""
        elif not isinstance(subpath, str):
            raise ValueError("Subpath must be a string.")

        self.subpath = subpath.strip()

        if customer_headers and not isinstance(customer_headers, Dict):
            raise ValueError("Customer headers must be a dictionary.")

        self.customer_headers = customer_headers

    @classmethod
    def class_name(cls) -> str:
        return "BAAIBGEEmbeddings"

    def _call_api(self, text: str) -> Embedding:
        headers = {"Content-Type": "application/json"}
        if self.customer_headers:
            headers.update(self.customer_headers)

        json_data = {"input": text, "model": self.model_name}

        with httpx.Client() as client:
            response = client.post(
                urljoin(self.base_url, self.subpath),
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return self._extract_embedding(response.json())

    async def _acall_api(self, text: str) -> Embedding:
        headers = {"Content-Type": "application/json"}
        if self.customer_headers:
            headers.update(self.customer_headers)

        json_data = {"input": text, "model": self.model_name}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                urljoin(self.base_url, self.subpath),
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return self._extract_embedding(response.json())

    def _extract_embedding(self, data: List[float] | Dict) -> Embedding:
        if isinstance(data, List):
            if data and not isinstance(data[0], float):
                raise ValueError("Invalid response format.")

            return data
        elif isinstance(data, Dict):
            # parse data here
            return data["data"][0]["embedding"]
        else:
            raise ValueError("Invalid response format.")

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        query = format_query(query, self.model_name)
        return self._call_api(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get text embedding."""
        text = format_text(text, self.model_name)
        return self._call_api(text)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding async."""
        query = format_query(query, self.model_name)
        return await self._acall_api(query)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Get text embedding async."""
        text = format_text(text, self.model_name)
        return await self._acall_api(text)
