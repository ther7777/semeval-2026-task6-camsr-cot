from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Optional

import aiohttp


@dataclass(frozen=True)
class ChatConfig:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_s: float
    max_retries: int
    concurrency: int


class OpenAIChatClient:
    """OpenAI-compatible Chat Completions 客户端（异步）。"""

    def __init__(self, *, api_key: str, cfg: ChatConfig):
        self._api_key = api_key
        self._cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(int(cfg.concurrency))
        self.api_call_count = 0  # 逻辑调用次数（按样本计数，不含内部重试）

    async def __aenter__(self) -> "OpenAIChatClient":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def chat(self, *, prompt: str) -> str:
        """发送一次 chat/completions 请求并返回模型输出文本。"""
        if self._session is None:
            raise RuntimeError("OpenAIChatClient 尚未进入上下文（请使用 async with）。")

        self.api_call_count += 1

        base_url = str(self._cfg.base_url).rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self._cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self._cfg.temperature),
            "max_tokens": int(self._cfg.max_tokens),
        }

        async with self._semaphore:
            for attempt in range(int(self._cfg.max_retries)):
                try:
                    async with self._session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=float(self._cfg.timeout_s),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return str(data["choices"][0]["message"]["content"])

                        # 常见可重试状态码
                        if resp.status in (429, 500, 502, 503, 504) and attempt < int(self._cfg.max_retries) - 1:
                            delay = min(60.0, (2**attempt) * 2.0) + random.uniform(0, 1)
                            await asyncio.sleep(delay)
                            continue

                        text = await resp.text()
                        raise RuntimeError(f"API error {resp.status}: {text[:200]}")

                except asyncio.TimeoutError:
                    if attempt < int(self._cfg.max_retries) - 1:
                        await asyncio.sleep(min(30.0, (2**attempt) * 2.0))
                        continue
                    raise
                except aiohttp.ClientError:
                    if attempt < int(self._cfg.max_retries) - 1:
                        await asyncio.sleep(min(30.0, (2**attempt) * 2.0))
                        continue
                    raise

        return ""

