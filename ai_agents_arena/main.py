#!/usr/bin/env python3
"""
main.py — AI Agents Arena
─────────────────────────────────────────────────────────────────────────────
Single entry point. Run all 8 agent types in competition mode.
Supports: OpenRouter (cloud) or llama-cpp-python (local).
Hardware-aware: auto-picks model size based on GPU/VPU/CPU.

Usage:
  python main.py                        # full competition, openrouter
  python main.py --backend local        # local GGUF inference
  python main.py --task coding          # single task only
  python main.py --agent lrm            # single agent type
  python main.py --list-tasks           # show all tasks
  python main.py --show-hw              # hardware detection only
  python main.py --demo                 # quick 1-task demo
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys

# Make imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

from config import OPENROUTER_API_KEY, COMPETITION_TASKS, OPENROUTER_MODELS
from hardware_detect import detect_hardware, print_hardware_report


def check_api_key():
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("\n⚠️  OpenRouter API key not set!")
        print("   Set it via environment variable:")
        print("   export OPENROUTER_API_KEY=sk-or-v1-your-key-here\n")
        print("   Or edit config.py → OPENROUTER_API_KEY\n")
        print("   Get a free key at: https://openrouter.ai/keys\n")
        return False
    return True


def run_demo():
    """Quick demo: run 1 task with all 8 agents."""
    from competition.arena import AgentArena
    task = {
        "id":     "demo",
        "name":   "Quick Demo",
        "prompt": "What is the Transformer architecture and why is it important for AI agents? "
                  "Answer in 2-3 sentences.",
    }
    arena = AgentArena(backend="openrouter", parallel=False)
    arena.run_task(task)
    arena.print_leaderboard()


def run_single_agent(agent_type: str, prompt: str):
    """Run a specific agent type with a custom prompt."""
    from agents.all_agents import (
        LLMAgent, MoEAgent, LRMAgent, VLMAgent,
        SLMAgent, AgenticAgent, OpenSourceAgent, SpecializedAgent,
    )
    m = OPENROUTER_MODELS
    AGENT_MAP = {
        "llm":        LLMAgent("LLM Agent",        m["llm"],        backend="openrouter"),
        "moe":        MoEAgent("MoE Agent",         m["moe"],        backend="openrouter"),
        "lrm":        LRMAgent("LRM Agent",         m["lrm"],        backend="openrouter"),
        "vlm":        VLMAgent("VLM Agent",         m["vlm"],        backend="openrouter"),
        "slm":        SLMAgent("SLM Agent",         m["slm"],        backend="openrouter"),
        "agentic":    AgenticAgent("Agentic Agent", m["agentic"],    backend="openrouter"),
        "opensource": OpenSourceAgent("OSS Agent",  m["opensource"], backend="openrouter"),
        "specialized":SpecializedAgent("Specialist",m["specialized"],backend="openrouter", domain="coding"),
    }
    agent = AGENT_MAP.get(agent_type.lower())
    if not agent:
        print(f"Unknown agent type: {agent_type}")
        print(f"Available: {list(AGENT_MAP.keys())}")
        return

    print(f"\n🤖 Running {agent_type.upper()} agent...")
    resp = agent.run(prompt)
    print(resp.pretty())


def main():
    parser = argparse.ArgumentParser(
        description="🏟️  AI Agents Arena — 8 Agent Types Competition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Full competition (OpenRouter)
  python main.py --demo                   Quick 1-task demo
  python main.py --show-hw                Hardware detection report
  python main.py --task coding            Single task
  python main.py --agent lrm --prompt "Prove sqrt(2) is irrational"
  python main.py --list-tasks             List all available tasks
  python main.py --backend local          Use local GGUF models
  python main.py --parallel               Run agents in parallel (faster)
        """
    )

    parser.add_argument("--backend",    default="openrouter", choices=["openrouter", "local"],
                        help="Inference backend")
    parser.add_argument("--parallel",   action="store_true",
                        help="Run agents in parallel threads")
    parser.add_argument("--task",       default=None,
                        help="Run single task by ID")
    parser.add_argument("--agent",      default=None,
                        help="Run single agent type (llm|moe|lrm|vlm|slm|agentic|opensource|specialized)")
    parser.add_argument("--prompt",     default=None,
                        help="Custom prompt (used with --agent)")
    parser.add_argument("--demo",       action="store_true",
                        help="Quick demo mode (1 task)")
    parser.add_argument("--show-hw",    action="store_true",
                        help="Show hardware detection report and exit")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List all competition tasks")

    args = parser.parse_args()

    # ── Hardware report ───────────────────────────────────────────────────────
    if args.show_hw:
        profile = detect_hardware()
        print_hardware_report(profile)
        print("Model suggestions for your hardware:")
        for agent_type, model in profile.model_suggestions.items():
            print(f"  {agent_type:<15} → {model}")
        return

    # ── List tasks ────────────────────────────────────────────────────────────
    if args.list_tasks:
        print("\n📋 Available competition tasks:\n")
        for t in COMPETITION_TASKS:
            print(f"  [{t['id']:<12}] {t['name']}")
            print(f"              {t['prompt'][:70]}...\n")
        return

    # ── Demo mode ─────────────────────────────────────────────────────────────
    if args.demo:
        if args.backend == "openrouter" and not check_api_key():
            return
        run_demo()
        return

    # ── Single agent ──────────────────────────────────────────────────────────
    if args.agent:
        if args.backend == "openrouter" and not check_api_key():
            return
        prompt = args.prompt or "Explain your architecture and what makes you unique as an AI agent."
        run_single_agent(args.agent, prompt)
        return

    # ── Full competition ──────────────────────────────────────────────────────
    if args.backend == "openrouter" and not check_api_key():
        return

    from competition.arena import AgentArena
    arena = AgentArena(backend=args.backend, parallel=args.parallel)

    if args.task:
        task = next((t for t in COMPETITION_TASKS if t["id"] == args.task), None)
        if not task:
            print(f"Task '{args.task}' not found. Use --list-tasks to see options.")
            return
        arena.run_task(task)
        arena.print_leaderboard()
    else:
        arena.run_all()


if __name__ == "__main__":
    main()
