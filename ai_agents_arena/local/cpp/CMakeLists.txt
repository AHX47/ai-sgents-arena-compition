cmake_minimum_required(VERSION 3.16)
project(ai_agents_arena_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ───────────────────────────────────────────────────────────────────
option(GGML_CUDA   "Enable NVIDIA CUDA GPU acceleration" OFF)
option(GGML_METAL  "Enable Apple Metal GPU acceleration" OFF)
option(GGML_VULKAN "Enable Vulkan GPU acceleration"      OFF)

# ── llama.cpp submodule ───────────────────────────────────────────────────────
# Clone llama.cpp into local/cpp/llama.cpp:
#   cd local/cpp && git clone https://github.com/ggerganov/llama.cpp
add_subdirectory(llama.cpp)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(llm_inference llm_inference.cpp)

target_include_directories(llm_inference PRIVATE
    llama.cpp/include
    llama.cpp/ggml/include
)

target_link_libraries(llm_inference PRIVATE llama ggml)

# ── Build flags ───────────────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(llm_inference PRIVATE -O3 -march=native)
endif()

# ── CUDA ──────────────────────────────────────────────────────────────────────
if(GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(llm_inference PRIVATE GGML_USE_CUDA)
    message(STATUS "CUDA acceleration enabled")
endif()

# ── Metal ─────────────────────────────────────────────────────────────────────
if(GGML_METAL)
    target_compile_definitions(llm_inference PRIVATE GGML_USE_METAL)
    message(STATUS "Metal acceleration enabled")
endif()

# ── Print build summary ───────────────────────────────────────────────────────
message(STATUS "═══════════════════════════════════════")
message(STATUS "  AI Agents Arena — C++ Local Inference")
message(STATUS "  CUDA:   ${GGML_CUDA}")
message(STATUS "  Metal:  ${GGML_METAL}")
message(STATUS "  Vulkan: ${GGML_VULKAN}")
message(STATUS "═══════════════════════════════════════")
