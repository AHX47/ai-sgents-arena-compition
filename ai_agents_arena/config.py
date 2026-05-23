"""
config.py — Central config. Set your OpenRouter API key here
or via environment variable OPENROUTER_API_KEY.
"""

import os

# ── OpenRouter (cloud — all 8 agent types) ────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_SITE_URL = "https://ai-agents-arena.local"
OPENROUTER_APP_NAME = "AI Agents Arena"

# ── OpenRouter model map (API names) ─────────────────────────────────────────
OPENROUTER_MODELS = {
    "llm":         "openai/gpt-4o-mini",               # fast LLM
    "moe":         "deepseek/deepseek-chat",            # DeepSeek V3 = MoE
    "lrm":         "deepseek/deepseek-r1",              # reasoning chain
    "vlm":         "google/gemini-flash-1.5",           # vision + text
    "slm":         "microsoft/phi-3-mini-128k-instruct",# 3.8B SLM
    "agentic":     "openai/gpt-4o-mini",               # tool calling
    "opensource":  "meta-llama/llama-3.3-70b-instruct",# open-weight frontier
    "specialized": "anthropic/claude-3-haiku",          # domain expert
}

# ── Local inference models (GGUF from HuggingFace) ───────────────────────────
# Download with: huggingface-cli download <repo> <filename> --local-dir ./models/
LOCAL_MODELS_GGUF = {
    "large": {
        "llm":    "models/llama-3.3-70b-instruct.Q4_K_M.gguf",
        "slm":    "models/phi-4.Q4_K_M.gguf",
        "lrm":    "models/deepseek-r1-distill-llama-70b.Q4_K_M.gguf",
    },
    "medium": {
        "llm":    "models/llama-3.1-8b-instruct.Q4_K_M.gguf",
        "slm":    "models/phi-3.5-mini-instruct.Q4_K_M.gguf",
        "lrm":    "models/deepseek-r1-distill-llama-8b.Q4_K_M.gguf",
    },
    "small": {
        "llm":    "models/gemma-2-2b-it.Q4_K_M.gguf",
        "slm":    "models/smollm2-1.7b-instruct.Q4_K_M.gguf",
        "lrm":    "models/deepseek-r1-distill-qwen-1.5b.Q4_K_M.gguf",
    },
    "tiny": {
        "llm":    "models/smollm2-360m-instruct.Q4_K_M.gguf",
        "slm":    "models/smollm2-360m-instruct.Q4_K_M.gguf",
        "lrm":    "models/deepseek-r1-distill-qwen-1.5b.Q4_K_M.gguf",
    },
}

# ── Scoring weights for competition ──────────────────────────────────────────
SCORING_WEIGHTS = {
    "quality":   0.40,
    "speed":     0.25,
    "accuracy":  0.25,
    "cost":      0.10,
}

# ── Default competition tasks ─────────────────────────────────────────────────
COMPETITION_TASKS = [
    {
        "id":   "reasoning",
        "name": "Multi-step Reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 die. How many are left? "
                  "Explain step by step.",
    },
    {
        "id":   "coding",
        "name": "Code Generation",
        "prompt": "Write a Python function that finds all prime numbers up to N "
                  "using the Sieve of Eratosthenes. Include type hints and docstring.",
    },
    {
        "id":   "summarize",
        "name": "Text Summarization",
        "prompt": "Summarize the concept of Mixture of Experts (MoE) in AI in 3 bullet points, "
                  "focusing on how routing works.",
    },
    {
        "id":   "creative",
        "name": "Creative Writing",
        "prompt": "Write a 4-line poem about an AI agent discovering consciousness.",
    },
    {
        "id":   "analysis",
        "name": "Data Analysis",
        "prompt": "Given sales data [120, 145, 98, 167, 203, 189, 156], "
                  "calculate mean, median, and identify the trend.",
    },
]
