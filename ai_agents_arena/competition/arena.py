"""
competition/arena.py
─────────────────────────────────────────────────────────────────
🏟️  AI AGENTS ARENA
Runs all 8 agent types on the same tasks, scores them, prints
a leaderboard.  Uses OpenRouter for cloud + hardware-aware local.
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import os
import sys
import time
import statistics
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.all_agents import (
    LLMAgent, MoEAgent, LRMAgent, VLMAgent,
    SLMAgent, AgenticAgent, OpenSourceAgent, SpecializedAgent,
)
from agents.base import AgentResponse
from config import OPENROUTER_MODELS, COMPETITION_TASKS, SCORING_WEIGHTS
from hardware_detect import detect_hardware, print_hardware_report


# ── Build the arena contestants ───────────────────────────────────────────────
def build_contestants(backend: str = "openrouter", hw_tier: str = "tiny") -> list:
    m = OPENROUTER_MODELS
    return [
        LLMAgent(
            name="🔵 LLM Classic",
            model=m["llm"],
            backend=backend,
            system_prompt="You are a helpful, accurate AI assistant.",
        ),
        MoEAgent(
            name="🔴 MoE Router",
            model=m["moe"],
            backend=backend,
        ),
        LRMAgent(
            name="🟡 Reasoner (LRM)",
            model=m["lrm"],
            backend=backend,
        ),
        VLMAgent(
            name="🟢 Vision-LM",
            model=m["vlm"],
            backend=backend,
        ),
        SLMAgent(
            name="🟣 SLM (Edge)",
            model=m["slm"],
            backend=backend,
        ),
        AgenticAgent(
            name="🔴 Agentic Actor",
            model=m["agentic"],
            backend=backend,
        ),
        OpenSourceAgent(
            name="🔵 Open-Source Frontier",
            model=m["opensource"],
            backend=backend,
            system_prompt="You are a powerful open-source AI assistant.",
        ),
        SpecializedAgent(
            name="🟢 Domain Specialist",
            model=m["specialized"],
            backend=backend,
            domain="coding",
        ),
    ]


# ── Naive quality scorer (no external LLM needed) ────────────────────────────
def _score_quality(response: str, task_id: str) -> float:
    """Very basic heuristic quality score 0–100."""
    if not response or len(response.strip()) < 10:
        return 0.0
    length_score = min(len(response) / 500, 1.0) * 30    # up to 30 pts for length
    has_structure = 10 if any(c in response for c in ["•", "-", "1.", "\n"]) else 0
    has_code = 20 if "```" in response or "def " in response or "function" in response else 0
    completeness = 40  # assume complete unless truncated
    if response.endswith("...") or len(response) < 50:
        completeness = 10
    return min(length_score + has_structure + has_code + completeness, 100)


def _score_response(resp: AgentResponse, task_id: str) -> dict[str, float]:
    quality  = _score_quality(resp.response, task_id)
    speed    = min(resp.score_speed / 50 * 100, 100) if resp.score_speed else 50  # 50 tok/s = 100
    accuracy = 80 if not resp.error else 0   # placeholder
    cost     = max(0, 100 - resp.cost_usd * 10000)   # lower cost = higher score

    w = SCORING_WEIGHTS
    total = (quality  * w["quality"]  +
             speed    * w["speed"]    +
             accuracy * w["accuracy"] +
             cost     * w["cost"])
    return {
        "quality": quality, "speed": speed,
        "accuracy": accuracy, "cost": cost,
        "total": total,
    }


# ── Arena runner ──────────────────────────────────────────────────────────────
class AgentArena:
    def __init__(self, backend: str = "openrouter", parallel: bool = False):
        self.backend    = backend
        self.parallel   = parallel
        self.hw_profile = detect_hardware()
        self.agents     = build_contestants(backend, self.hw_profile.recommended_tier)
        self.results: dict[str, list[AgentResponse]] = {}
        self.scores:  dict[str, list[float]] = {}

    def run_task(self, task: dict) -> None:
        print(f"\n{'═'*62}")
        print(f"  📋 TASK: {task['name']}")
        print(f"  ❓ {task['prompt'][:80]}...")
        print(f"{'═'*62}")

        task_id   = task["id"]
        task_resp = []

        if self.parallel:
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = {ex.submit(agent.run, task["prompt"]): agent for agent in self.agents}
                for fut in as_completed(futures):
                    resp = fut.result()
                    task_resp.append(resp)
                    print(resp.pretty())
        else:
            for agent in self.agents:
                print(f"  ⏳ Running {agent.name}...", end="", flush=True)
                resp = agent.run(task["prompt"])
                task_resp.append(resp)
                status = "✅" if not resp.error else "❌"
                print(f"\r  {status} {agent.name} — {resp.elapsed_sec:.2f}s")

        self.results[task_id] = task_resp

        # Score
        for resp in task_resp:
            scores = _score_response(resp, task_id)
            key    = resp.agent_name
            if key not in self.scores:
                self.scores[key] = []
            self.scores[key].append(scores["total"])

    def run_all(self, tasks: Optional[list] = None) -> None:
        tasks = tasks or COMPETITION_TASKS
        print_hardware_report(self.hw_profile)
        print(f"\n🏟️  AI AGENTS ARENA — {len(self.agents)} contestants × {len(tasks)} tasks\n")

        for task in tasks:
            self.run_task(task)
            time.sleep(1)    # be polite to the API

        self.print_leaderboard()

    def print_leaderboard(self) -> None:
        print(f"\n\n{'═'*62}")
        print("  🏆  FINAL LEADERBOARD")
        print(f"{'═'*62}")
        print(f"  {'RANK':<5} {'AGENT':<30} {'AVG SCORE':>10} {'TASKS':>6}")
        print(f"  {'─'*55}")

        ranking = []
        for agent_name, scores in self.scores.items():
            avg = statistics.mean(scores) if scores else 0
            ranking.append((agent_name, avg, len(scores)))

        ranking.sort(key=lambda x: x[1], reverse=True)
        medals = ["🥇", "🥈", "🥉"] + ["  "] * 10

        for i, (name, avg, n) in enumerate(ranking):
            medal = medals[i]
            print(f"  {medal}  {i+1:<3} {name:<30} {avg:>9.1f}   {n:>4}")

        print(f"{'═'*62}")
        winner = ranking[0][0] if ranking else "N/A"
        print(f"\n  🎉 Winner: {winner}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Agents Arena Competition")
    parser.add_argument("--backend",  default="openrouter", choices=["openrouter", "local"])
    parser.add_argument("--parallel", action="store_true", help="Run agents in parallel")
    parser.add_argument("--task",     default=None, help="Run single task by ID")
    args = parser.parse_args()

    arena = AgentArena(backend=args.backend, parallel=args.parallel)

    if args.task:
        task = next((t for t in COMPETITION_TASKS if t["id"] == args.task), None)
        if task:
            arena.run_task(task)
        else:
            print(f"Task '{args.task}' not found. Options: {[t['id'] for t in COMPETITION_TASKS]}")
    else:
        arena.run_all()
