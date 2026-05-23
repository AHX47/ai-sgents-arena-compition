"""
agents/all_agents.py
─────────────────────────────────────────────────────────────
8 agent architectures from the infographic, each as a class:

  1. LLMAgent          — standard autoregressive LLM
  2. MoEAgent          — Mixture of Experts (routing)
  3. LRMAgent          — Large Reasoning Model (chain-of-thought)
  4. VLMAgent          — Vision Language Model
  5. SLMAgent          — Small Language Model
  6. AgenticAgent      — Tool-calling / Action Model
  7. OpenSourceAgent   — Open-weight Frontier Model
  8. SpecializedAgent  — Domain-specific expert
"""

from __future__ import annotations
import base64
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.base import BaseAgent, AgentResponse
from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY, OPENROUTER_SITE_URL, OPENROUTER_APP_NAME
import requests
import time


# ══════════════════════════════════════════════════════════════════
# 1. LLM Agent  ─ classic transformer, token-by-token prediction
# ══════════════════════════════════════════════════════════════════
class LLMAgent(BaseAgent):
    """
    Architecture: Input → Tokenization → Embedding
                  → Multi-Layer Transformer → Sample next token → Output

    Best for: general-purpose tasks, chat, Q&A.
    Cloud example : GPT-4o, Claude, Gemini
    Local  example: Llama-3, Mistral
    """
    agent_type = "llm"

    def _build_messages(self, prompt: str, image_path=None) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]

    def _build_body(self, messages) -> dict:
        return {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }


# ══════════════════════════════════════════════════════════════════
# 2. MoE Agent  ─ Mixture of Experts with explicit routing logic
# ══════════════════════════════════════════════════════════════════
class MoEAgent(BaseAgent):
    """
    Architecture: Input → Tokenization → Gating Network
                  → Select Top-K Experts → Weighted Combination → Output

    Routing is done here in Python (client-side) mirroring how
    MoE models internally route tokens to specialized sub-networks.

    Cloud: DeepSeek V3 (MoE natively)
    Local: Mixtral (8 expert groups)
    """
    agent_type = "moe"

    # Domain experts — each with its own system prompt
    EXPERTS: dict[str, str] = {
        "coding":  "You are a senior software engineer. Write clean, production-ready code with comments.",
        "math":    "You are a mathematics professor. Show all working steps and verify your answer.",
        "writing": "You are a literary editor. Write with clarity, style, and engaging structure.",
        "science": "You are a research scientist. Be precise, cite principles, and reason empirically.",
        "general": "You are a knowledgeable generalist. Be concise and accurate.",
    }

    ROUTING_KEYWORDS: dict[str, list[str]] = {
        "coding":  ["code", "function", "bug", "python", "javascript", "algorithm", "program", "class", "def "],
        "math":    ["calculate", "solve", "equation", "integral", "derivative", "proof", "formula", "mathematics"],
        "writing": ["write", "essay", "poem", "story", "paragraph", "summarize", "draft", "creative"],
        "science": ["physics", "chemistry", "biology", "research", "experiment", "theory", "quantum"],
    }

    def _gate(self, prompt: str) -> str:
        """Client-side gating: select best expert for this prompt."""
        low = prompt.lower()
        scores = {domain: 0 for domain in self.EXPERTS}
        for domain, keywords in self.ROUTING_KEYWORDS.items():
            for kw in keywords:
                if kw in low:
                    scores[domain] += 1
        best = max(scores, key=lambda d: scores[d])
        return best if scores[best] > 0 else "general"

    def run(self, prompt: str, image_path=None) -> AgentResponse:
        t0           = time.perf_counter()
        expert_name  = self._gate(prompt)
        expert_sys   = self.EXPERTS[expert_name]

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer":  OPENROUTER_SITE_URL,
            "X-Title":       OPENROUTER_APP_NAME,
            "Content-Type":  "application/json",
        }
        body = {
            "model":      self.model,
            "messages":   [
                {"role": "system", "content": expert_sys},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            resp = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions",
                                 headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            data  = resp.json()
            usage = data.get("usage", {})
            return AgentResponse(
                agent_name = f"{self.name} [expert:{expert_name}]",
                agent_type = self.agent_type,
                prompt     = prompt,
                response   = data["choices"][0]["message"]["content"],
                tokens_in  = usage.get("prompt_tokens", 0),
                tokens_out = usage.get("completion_tokens", 0),
                elapsed_sec= time.perf_counter() - t0,
                backend    = "openrouter",
                model      = self.model,
            )
        except Exception as e:
            return AgentResponse(agent_name=self.name, agent_type=self.agent_type,
                                 prompt=prompt, response="", error=str(e),
                                 elapsed_sec=time.perf_counter()-t0, model=self.model, backend="openrouter")

    def _build_messages(self, prompt, image_path=None): return []
    def _build_body(self, messages): return {}


# ══════════════════════════════════════════════════════════════════
# 3. LRM Agent  ─ Large Reasoning Model (chain-of-thought)
# ══════════════════════════════════════════════════════════════════
class LRMAgent(BaseAgent):
    """
    Architecture: Input → Break down problem → Explore options (1→2→3)
                  → Check logic → Final Answer

    Forces the model into explicit <think> / reasoning mode.
    Cloud: DeepSeek-R1, o1, Claude with extended thinking
    """
    agent_type = "lrm"

    # Prompt template that forces chain-of-thought
    REASONING_TEMPLATE = """\
Think step by step. Break the problem into parts, explore options, \
check your logic, then give a FINAL ANSWER clearly marked.

Problem: {prompt}

Format:
<thinking>
[your full reasoning here]
</thinking>
<answer>
[final concise answer]
</answer>"""

    def _build_messages(self, prompt: str, image_path=None) -> list[dict]:
        return [
            {"role": "system", "content": "You are a deep reasoner. Always think before answering."},
            {"role": "user",   "content": self.REASONING_TEMPLATE.format(prompt=prompt)},
        ]

    def _build_body(self, messages) -> dict:
        return {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": 0.1,    # low temp for reasoning
        }

    def _run_openrouter(self, prompt: str, image_path=None) -> AgentResponse:
        result = super()._run_openrouter(prompt, image_path)
        # Parse thinking from response
        raw = result.response
        thinking, answer = "", raw
        if "<thinking>" in raw and "</thinking>" in raw:
            thinking = raw.split("<thinking>")[1].split("</thinking>")[0].strip()
        if "<answer>" in raw and "</answer>" in raw:
            answer = raw.split("<answer>")[1].split("</answer>")[0].strip()
        result.thinking = thinking
        result.response = answer
        return result


# ══════════════════════════════════════════════════════════════════
# 4. VLM Agent  ─ Vision Language Model
# ══════════════════════════════════════════════════════════════════
class VLMAgent(BaseAgent):
    """
    Architecture: Image → Vision Encoder → Embedding
                  Text  → Tokenization   → Embedding
                  → Merge → Unified Transformer → Final Answer

    Handles both text-only and image+text inputs.
    Cloud: GPT-4o, Gemini Flash, Qwen-VL
    Local: LLaVA, moondream2
    """
    agent_type = "vlm"

    def _build_messages(self, prompt: str, image_path: Optional[str] = None) -> list[dict]:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(image_path)[-1].lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "gif": "image/gif",
                    "webp": "image/webp"}.get(ext, "image/jpeg")
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        return [
            {"role": "system", "content": "You are a vision AI. Analyze images and text together."},
            {"role": "user",   "content": content},
        ]

    def _build_body(self, messages) -> dict:
        return {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }


# ══════════════════════════════════════════════════════════════════
# 5. SLM Agent  ─ Small Language Model
# ══════════════════════════════════════════════════════════════════
class SLMAgent(BaseAgent):
    """
    Architecture: Large Model → Knowledge Distillation → Small Model
                  → Compact Transformer → Quantization → Efficient Inference

    Designed for edge/offline use. Very fast on CPU.
    Cloud fallback: Phi-3-mini (~3.8B), Gemma-2 2B
    Local: GGUF quantized (Q4_K_M) on CPU
    """
    agent_type = "slm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # SLM defaults: shorter context, higher temp for diversity
        self.max_tokens  = min(self.max_tokens, 512)
        self.temperature = 0.8

    def _build_messages(self, prompt: str, image_path=None) -> list[dict]:
        # SLMs work best with very concise system prompts
        return [
            {"role": "system", "content": "Be brief and accurate."},
            {"role": "user",   "content": prompt},
        ]

    def _build_body(self, messages) -> dict:
        return {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }


# ══════════════════════════════════════════════════════════════════
# 6. Agentic (Action) Agent  ─ Plan → Tools → Execute → Check
# ══════════════════════════════════════════════════════════════════
class AgenticAgent(BaseAgent):
    """
    Architecture: Input → Understand Intent → Plan Steps
                  → Use Tools/APIs → Execute Actions
                  → Check Result → Output

    Implements a ReAct-style loop (Reason + Act).
    Includes simulated tools: calculator, web_search, run_code.
    """
    agent_type = "agentic"

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression. Use for arithmetic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression, e.g. '(12 * 3) / 4'"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search for current information on the web.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Execute a Python code snippet and return the output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to run"},
                    },
                    "required": ["code"],
                },
            },
        },
    ]

    # ── Simulated tool executors ──────────────────────────────────────────────
    @staticmethod
    def _exec_tool(name: str, args: dict) -> str:
        if name == "calculator":
            try:
                result = eval(args["expression"], {"__builtins__": {}})
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
        elif name == "web_search":
            return f"[SIMULATED] Search results for '{args['query']}': No live results (demo mode)."
        elif name == "run_python":
            import io, contextlib
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(args["code"], {})  # noqa: S102
                return buf.getvalue() or "(no output)"
            except Exception as e:
                return f"Error: {e}"
        return f"Unknown tool: {name}"

    def run(self, prompt: str, image_path=None) -> AgentResponse:
        t0         = time.perf_counter()
        tools_used = []
        messages   = [
            {"role": "system", "content": (
                "You are an action-taking AI agent. For each task: "
                "1) Understand intent  2) Plan steps  3) Use tools if needed  "
                "4) Execute  5) Check result  6) Return final answer."
            )},
            {"role": "user", "content": prompt},
        ]
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer":  OPENROUTER_SITE_URL,
            "X-Title":       OPENROUTER_APP_NAME,
            "Content-Type":  "application/json",
        }
        final_response = ""

        for _step in range(5):     # max 5 tool-call rounds
            body = {
                "model":       self.model,
                "messages":    messages,
                "tools":       self.TOOLS,
                "max_tokens":  self.max_tokens,
                "temperature": self.temperature,
            }
            try:
                resp = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions",
                                     headers=headers, json=body, timeout=120)
                resp.raise_for_status()
            except Exception as e:
                return AgentResponse(agent_name=self.name, agent_type=self.agent_type,
                                     prompt=prompt, response="", error=str(e),
                                     elapsed_sec=time.perf_counter()-t0, model=self.model, backend="openrouter")

            data    = resp.json()
            choice  = data["choices"][0]
            message = choice["message"]
            messages.append(message)

            if choice.get("finish_reason") == "tool_calls" and message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    tool_result = self._exec_tool(fn_name, fn_args)
                    tools_used.append(fn_name)
                    messages.append({
                        "role":       "tool",
                        "tool_call_id": tc["id"],
                        "content":    tool_result,
                    })
            else:
                final_response = message.get("content") or ""
                break

        usage = data.get("usage", {})
        return AgentResponse(
            agent_name  = self.name,
            agent_type  = self.agent_type,
            prompt      = prompt,
            response    = final_response,
            tools_used  = tools_used,
            tokens_in   = usage.get("prompt_tokens", 0),
            tokens_out  = usage.get("completion_tokens", 0),
            elapsed_sec = time.perf_counter() - t0,
            backend     = "openrouter",
            model       = self.model,
        )

    def _build_messages(self, prompt, image_path=None): return []
    def _build_body(self, messages): return {}


# ══════════════════════════════════════════════════════════════════
# 7. Open-Source Frontier Agent
# ══════════════════════════════════════════════════════════════════
class OpenSourceAgent(BaseAgent):
    """
    Architecture: Input → Open-weight model → Customize / Fine-tune
                  → Deploy self-hosted or hybrid → Output

    Uses fully open-weight models. Can be fine-tuned locally.
    Cloud fallback: Llama-3.3-70B, Qwen-2.5-72B via OpenRouter
    Local: any GGUF on llama.cpp, or HuggingFace Transformers
    """
    agent_type = "opensource"

    def _build_messages(self, prompt: str, image_path=None) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]

    def _build_body(self, messages) -> dict:
        return {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }

    def fine_tune_example(self) -> str:
        """Returns HuggingFace fine-tuning code snippet."""
        return '''\
# Fine-tune with HuggingFace TRL (QLoRA)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

# 4-bit quantization (fits on 8GB GPU)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model_id = "meta-llama/Llama-3.1-8B-Instruct"
model    = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_cfg)
tok      = AutoTokenizer.from_pretrained(model_id)

# LoRA config
lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])
model    = get_peft_model(model, lora_cfg)

# Train
trainer = SFTTrainer(
    model=model, tokenizer=tok,
    train_dataset=your_dataset,
    args=SFTConfig(output_dir="./my-llama", num_train_epochs=3),
)
trainer.train()
trainer.save_model("./my-llama-finetuned")
'''


# ══════════════════════════════════════════════════════════════════
# 8. Specialized Domain Agent
# ══════════════════════════════════════════════════════════════════
class SpecializedAgent(BaseAgent):
    """
    Architecture: Input → Domain Understanding
                  → Task-specific Processing → Specialized Output

    Uses domain-expert system prompts + structured output formatting.
    Works best with: Claude 3 Haiku, GPT-4o-mini, fine-tuned models.
    """
    agent_type = "specialized"

    DOMAIN_CONFIGS: dict[str, dict] = {
        "medical": {
            "system": (
                "You are a clinical AI assistant trained on medical literature. "
                "Always recommend professional consultation. Structure: "
                "Symptoms | Differential Diagnosis | Recommended Tests | Caution."
            ),
            "temperature": 0.1,
        },
        "legal": {
            "system": (
                "You are a legal AI assistant. Cite applicable laws/precedents. "
                "Disclaimer: not a substitute for licensed legal advice. "
                "Structure: Legal Issue | Applicable Law | Analysis | Recommendation."
            ),
            "temperature": 0.1,
        },
        "finance": {
            "system": (
                "You are a quantitative finance AI. Use data-driven analysis. "
                "Disclaimer: not financial advice. "
                "Structure: Market Context | Technical Analysis | Risk Factors | Outlook."
            ),
            "temperature": 0.2,
        },
        "coding": {
            "system": (
                "You are a senior software engineer AI. Return ONLY working, "
                "tested code with type hints, docstrings, and edge-case handling."
            ),
            "temperature": 0.1,
        },
        "education": {
            "system": (
                "You are an expert teacher AI. Explain concepts from first principles. "
                "Use analogies, examples, and check understanding with a quiz question."
            ),
            "temperature": 0.7,
        },
    }

    def __init__(self, *args, domain: str = "coding", **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = domain
        cfg = self.DOMAIN_CONFIGS.get(domain, {})
        if cfg.get("system"):
            self.system_prompt = cfg["system"]
        if cfg.get("temperature") is not None:
            self.temperature = cfg["temperature"]

    def _build_messages(self, prompt: str, image_path=None) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]

    def _build_body(self, messages) -> dict:
        return {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }
