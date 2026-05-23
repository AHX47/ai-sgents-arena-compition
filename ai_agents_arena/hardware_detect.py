"""
hardware_detect.py — Auto-detects GPU / VPU / CPU
and picks the right model tier for local inference.
"""

import subprocess
import platform
import sys
import os
from dataclasses import dataclass
from typing import Literal


HardwareType = Literal["cuda_gpu", "rocm_gpu", "intel_vpu", "apple_silicon", "cpu"]


@dataclass
class HardwareProfile:
    hw_type: HardwareType
    device_name: str
    vram_gb: float          # 0 if CPU
    recommended_tier: Literal["large", "medium", "small", "tiny"]
    local_backend: Literal["llama_cpp_cuda", "llama_cpp_metal", "llama_cpp_cpu", "openvino"]
    model_suggestions: dict  # agent_type -> model_name


# ─────────────────────────────────────────────
# Model selection per tier
# ─────────────────────────────────────────────
MODELS_BY_TIER = {
    "large": {
        "llm":         "meta-llama/Llama-3.3-70B-Instruct",
        "moe":         "deepseek-ai/DeepSeek-V3",
        "lrm":         "deepseek-ai/DeepSeek-R1",
        "vlm":         "Qwen/Qwen2-VL-72B-Instruct",
        "slm":         "microsoft/Phi-4",
        "agentic":     "meta-llama/Llama-3.3-70B-Instruct",
        "opensource":  "deepseek-ai/DeepSeek-R1",
        "specialized": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "medium": {
        "llm":         "meta-llama/Llama-3.1-8B-Instruct",
        "moe":         "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "lrm":         "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "vlm":         "Qwen/Qwen2-VL-7B-Instruct",
        "slm":         "microsoft/Phi-3.5-mini-instruct",
        "agentic":     "meta-llama/Llama-3.1-8B-Instruct",
        "opensource":  "mistralai/Mistral-7B-Instruct-v0.3",
        "specialized": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "small": {
        "llm":         "google/gemma-2-2b-it",
        "moe":         "mistralai/Mistral-7B-Instruct-v0.3",
        "lrm":         "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "vlm":         "moondream/moondream2",
        "slm":         "google/gemma-2-2b-it",
        "agentic":     "microsoft/Phi-3.5-mini-instruct",
        "opensource":  "google/gemma-2-2b-it",
        "specialized": "microsoft/Phi-3.5-mini-instruct",
    },
    "tiny": {
        "llm":         "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "moe":         "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "lrm":         "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "vlm":         "moondream/moondream2",
        "slm":         "HuggingFaceTB/SmolLM2-360M-Instruct",
        "agentic":     "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "opensource":  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "specialized": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    },
}


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def detect_hardware() -> HardwareProfile:
    """Main entry — returns a HardwareProfile for the current machine."""

    # ── NVIDIA CUDA ──────────────────────────────────────────────────────────
    nvidia = _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if nvidia:
        parts  = nvidia.split(",")
        name   = parts[0].strip()
        vram   = float(parts[1].replace("MiB", "").strip()) / 1024 if len(parts) > 1 else 0
        tier   = "large" if vram >= 20 else "medium" if vram >= 8 else "small" if vram >= 4 else "tiny"
        return HardwareProfile(
            hw_type="cuda_gpu", device_name=name, vram_gb=vram,
            recommended_tier=tier, local_backend="llama_cpp_cuda",
            model_suggestions=MODELS_BY_TIER[tier],
        )

    # ── AMD ROCm ─────────────────────────────────────────────────────────────
    rocm = _run("rocm-smi --showmeminfo vram --csv")
    if rocm and "GPU" in rocm:
        return HardwareProfile(
            hw_type="rocm_gpu", device_name="AMD ROCm GPU", vram_gb=8,
            recommended_tier="medium", local_backend="llama_cpp_cuda",
            model_suggestions=MODELS_BY_TIER["medium"],
        )

    # ── Apple Silicon (Metal) ─────────────────────────────────────────────────
    if platform.system() == "Darwin":
        chip = _run("sysctl -n machdep.cpu.brand_string")
        if "Apple" in chip or "M1" in chip or "M2" in chip or "M3" in chip or "M4" in chip:
            ram_bytes = int(_run("sysctl -n hw.memsize") or "0")
            ram_gb    = ram_bytes / (1024**3)
            tier = "large" if ram_gb >= 32 else "medium" if ram_gb >= 16 else "small"
            return HardwareProfile(
                hw_type="apple_silicon", device_name=chip, vram_gb=ram_gb,
                recommended_tier=tier, local_backend="llama_cpp_metal",
                model_suggestions=MODELS_BY_TIER[tier],
            )

    # ── Intel VPU / NPU (OpenVINO) ───────────────────────────────────────────
    openvino = _run("python3 -c 'from openvino.runtime import Core; c=Core(); print(c.available_devices)'")
    if "NPU" in openvino or "VPU" in openvino:
        return HardwareProfile(
            hw_type="intel_vpu", device_name="Intel NPU/VPU", vram_gb=0,
            recommended_tier="small", local_backend="openvino",
            model_suggestions=MODELS_BY_TIER["small"],
        )

    # ── CPU fallback ─────────────────────────────────────────────────────────
    cpu_name = _run("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2").strip()
    if not cpu_name:
        cpu_name = platform.processor() or "Unknown CPU"
    return HardwareProfile(
        hw_type="cpu", device_name=cpu_name, vram_gb=0,
        recommended_tier="tiny", local_backend="llama_cpp_cpu",
        model_suggestions=MODELS_BY_TIER["tiny"],
    )


def print_hardware_report(profile: HardwareProfile) -> None:
    icons = {
        "cuda_gpu": "🟢 NVIDIA CUDA", "rocm_gpu": "🟠 AMD ROCm",
        "intel_vpu": "🔵 Intel NPU/VPU", "apple_silicon": "🍎 Apple Silicon",
        "cpu": "⚪ CPU Only",
    }
    print("\n" + "═"*50)
    print(f"  🔍 HARDWARE DETECTION REPORT")
    print("═"*50)
    print(f"  Type   : {icons[profile.hw_type]}")
    print(f"  Device : {profile.device_name}")
    if profile.vram_gb:
        print(f"  VRAM   : {profile.vram_gb:.1f} GB")
    print(f"  Tier   : {profile.recommended_tier.upper()}")
    print(f"  Backend: {profile.local_backend}")
    print("═"*50 + "\n")


if __name__ == "__main__":
    p = detect_hardware()
    print_hardware_report(p)
