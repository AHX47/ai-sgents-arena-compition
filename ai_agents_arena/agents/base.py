"""
agents/base.py — Abstract base for all 8 agent types.
Supports: OpenRouter API (cloud) + llama-cpp-python (local).
"""

from __future__ import annotations
import time
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_APP_NAME, OPENROUTER_SITE_URL


@dataclass
class AgentResponse:
    agent_name:    str
    agent_type:    str
    prompt:        str
    response:      str
    thinking:      Optional[str]   = None
    tools_used:    list[str]       = field(default_factory=list)
    elapsed_sec:   float           = 0.0
    tokens_in:     int             = 0
    tokens_out:    int             = 0
    cost_usd:      float           = 0.0
    backend:       str             = "openrouter"    # or "local"
    model:         str             = ""
    error:         Optional[str]   = None

    @property
    def score_speed(self) -> float:
        """Tokens per second (higher = better)."""
        if self.elapsed_sec == 0:
            return 0
        return self.tokens_out / self.elapsed_sec

    def pretty(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  🤖 {self.agent_name} [{self.agent_type.upper()}]",
            f"  Model   : {self.model}",
            f"  Backend : {self.backend}",
            f"  Time    : {self.elapsed_sec:.2f}s  |  {self.score_speed:.1f} tok/s",
            f"  Tokens  : ↑{self.tokens_in}  ↓{self.tokens_out}",
            f"  Cost    : ${self.cost_usd:.5f}",
        ]
        if self.thinking:
            lines.append(f"\n  🧠 THINKING (excerpt):\n  {self.thinking[:300]}...")
        if self.tools_used:
            lines.append(f"\n  🔧 Tools: {', '.join(self.tools_used)}")
        lines.append(f"\n  📝 RESPONSE:\n  {self.response}")
        lines.append("─"*60)
        return "\n".join(lines)


class BaseAgent(ABC):
    """Abstract base — subclass for each of the 8 agent architectures."""

    agent_type: str = "base"

    def __init__(
        self,
        name: str,
        model: str,
        backend: str = "openrouter",          # "openrouter" | "local"
        local_model_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.name             = name
        self.model            = model
        self.backend          = backend
        self.local_model_path = local_model_path
        self.system_prompt    = system_prompt or "You are a helpful AI assistant."
        self.temperature      = temperature
        self.max_tokens       = max_tokens
        self._llama           = None           # lazy-loaded local model

    # ── Public API ────────────────────────────────────────────────────────────
    def run(self, prompt: str, image_path: Optional[str] = None) -> AgentResponse:
        t0 = time.perf_counter()
        try:
            if self.backend == "local":
                result = self._run_local(prompt)
            else:
                result = self._run_openrouter(prompt, image_path)
        except Exception as e:
            return AgentResponse(
                agent_name=self.name, agent_type=self.agent_type,
                prompt=prompt, response="", error=str(e),
                elapsed_sec=time.perf_counter() - t0, model=self.model,
                backend=self.backend,
            )
        result.elapsed_sec = time.perf_counter() - t0
        return result

    # ── OpenRouter cloud call ─────────────────────────────────────────────────
    def _run_openrouter(
        self, prompt: str, image_path: Optional[str] = None
    ) -> AgentResponse:
        headers = {
            "Authorization":  f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer":   OPENROUTER_SITE_URL,
            "X-Title":        OPENROUTER_APP_NAME,
            "Content-Type":   "application/json",
        }
        messages = self._build_messages(prompt, image_path)
        body     = self._build_body(messages)

        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers, json=body, timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        choice      = data["choices"][0]
        content_raw = choice["message"].get("content") or ""
        usage       = data.get("usage", {})

        return AgentResponse(
            agent_name  = self.name,
            agent_type  = self.agent_type,
            prompt      = prompt,
            response    = content_raw,
            tokens_in   = usage.get("prompt_tokens", 0),
            tokens_out  = usage.get("completion_tokens", 0),
            backend     = "openrouter",
            model       = data.get("model", self.model),
        )

    # ── Local inference via llama-cpp-python ──────────────────────────────────
    def _run_local(self, prompt: str) -> AgentResponse:
        if self._llama is None:
            self._llama = self._load_local_model()
        output = self._llama.create_chat_completion(
            messages=[
                {"role": "system",  "content": self.system_prompt},
                {"role": "user",    "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        choice = output["choices"][0]
        usage  = output.get("usage", {})
        return AgentResponse(
            agent_name = self.name,
            agent_type = self.agent_type,
            prompt     = prompt,
            response   = choice["message"]["content"],
            tokens_in  = usage.get("prompt_tokens", 0),
            tokens_out = usage.get("completion_tokens", 0),
            backend    = "local",
            model      = self.local_model_path or self.model,
        )

    def _load_local_model(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Run:\n"
                "  pip install llama-cpp-python   (CPU)\n"
                "  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python  (CUDA)"
            )
        if not self.local_model_path or not os.path.exists(self.local_model_path):
            raise FileNotFoundError(
                f"GGUF model not found: {self.local_model_path}\n"
                "Download from HuggingFace with:\n"
                "  huggingface-cli download <repo> <file> --local-dir ./models/"
            )
        return Llama(
            model_path   = self.local_model_path,
            n_ctx        = 4096,
            n_gpu_layers = -1,     # auto: all layers to GPU if available
            verbose      = False,
        )

    # ── Override these in subclasses ──────────────────────────────────────────
    @abstractmethod
    def _build_messages(self, prompt: str, image_path: Optional[str]) -> list[dict]:
        ...

    @abstractmethod
    def _build_body(self, messages: list[dict]) -> dict:
        ...
