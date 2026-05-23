/**
 * local/cpp/llm_inference.cpp
 * ─────────────────────────────────────────────────────────────────
 * Local LLM inference using llama.cpp C API.
 * Auto-selects GPU (CUDA/Metal) or CPU based on available hardware.
 *
 * BUILD (CPU):
 *   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
 *   cmake -B build && cmake --build build --config Release -j$(nproc)
 *   g++ -std=c++17 -O3 -I llama.cpp/include -I llama.cpp/ggml/include \
 *       llm_inference.cpp -L llama.cpp/build/src -lllama -lggml \
 *       -Wl,-rpath,llama.cpp/build/src -o llm_inference
 *
 * BUILD (CUDA):
 *   cmake -B build -DGGML_CUDA=ON
 *   cmake --build build --config Release -j$(nproc)
 *   [same g++ command above]
 *
 * BUILD (Metal / Apple Silicon):
 *   cmake -B build -DGGML_METAL=ON
 *   cmake --build build --config Release
 *
 * RUN:
 *   ./llm_inference models/llama-3.1-8b-instruct.Q4_K_M.gguf \
 *       "Explain the difference between MoE and LLM architectures"
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "llama.h"       // from llama.cpp/include/

// ── Hardware detection ────────────────────────────────────────────────────────
struct HardwareInfo {
    bool has_cuda   = false;
    bool has_metal  = false;
    int  gpu_layers = 0;     // 0 = CPU, -1 = all layers on GPU
};

static HardwareInfo detect_hw() {
    HardwareInfo info;

#if defined(GGML_USE_CUDA)
    info.has_cuda   = true;
    info.gpu_layers = -1;   // offload everything to CUDA GPU
    fprintf(stderr, "[HW] CUDA GPU detected — offloading all layers\n");

#elif defined(GGML_USE_METAL)
    info.has_metal  = true;
    info.gpu_layers = -1;   // offload to Metal (Apple Silicon)
    fprintf(stderr, "[HW] Apple Metal detected — offloading all layers\n");

#else
    info.gpu_layers = 0;    // CPU only
    fprintf(stderr, "[HW] CPU only — using quantized model for efficiency\n");
#endif

    return info;
}

// ── Simple completion function ────────────────────────────────────────────────
struct CompletionResult {
    std::string text;
    int         tokens_generated = 0;
    double      elapsed_ms       = 0.0;
    double      tokens_per_sec   = 0.0;
};

CompletionResult run_completion(
    llama_model*   model,
    llama_context* ctx,
    const std::string& system_prompt,
    const std::string& user_prompt,
    int max_new_tokens = 512
) {
    CompletionResult result;

    // ── Build chat prompt (Llama-3 instruct format) ───────────────────────────
    std::string full_prompt =
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"   + user_prompt   + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    // ── Tokenize ──────────────────────────────────────────────────────────────
    const int n_ctx = llama_n_ctx(ctx);
    std::vector<llama_token> tokens(full_prompt.size() + 16);
    int n_tokens = llama_tokenize(
        llama_model_get_model(model),
        full_prompt.c_str(), full_prompt.size(),
        tokens.data(), tokens.size(),
        /*add_bos=*/false, /*special=*/true
    );
    if (n_tokens < 0) {
        fprintf(stderr, "Tokenization failed\n");
        return result;
    }
    tokens.resize(n_tokens);

    // ── Decode (prefill) ──────────────────────────────────────────────────────
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Decode failed\n");
        return result;
    }

    // ── Generate tokens ───────────────────────────────────────────────────────
    auto t_start = std::chrono::high_resolution_clock::now();

    llama_token eos_token = llama_token_eos(llama_model_get_model(model));
    std::string output;
    int         n_generated = 0;

    while (n_generated < max_new_tokens) {
        // Greedy sampling
        llama_token new_token = llama_sampler_sample(
            llama_sampler_chain_init(llama_sampler_chain_default_params()),
            ctx, -1
        );

        if (new_token == eos_token) break;

        // Token → text
        char buf[256] = {};
        int  n_chars  = llama_token_to_piece(
            llama_model_get_model(model), new_token, buf, sizeof(buf), 0, true
        );
        if (n_chars > 0) {
            output.append(buf, n_chars);
            printf("%.*s", n_chars, buf);
            fflush(stdout);
        }

        // Prepare next token batch
        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, next_batch)) break;

        n_generated++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    result.text             = output;
    result.tokens_generated = n_generated;
    result.elapsed_ms       = elapsed_ms;
    result.tokens_per_sec   = n_generated / (elapsed_ms / 1000.0);

    return result;
}

// ── 8 Agent type runners (each with its own system prompt) ───────────────────
void run_agent(
    llama_model*   model,
    llama_context* ctx,
    const std::string& agent_type,
    const std::string& user_prompt
) {
    static const struct { const char* type; const char* sys; } AGENTS[] = {
        {"LLM",        "You are a helpful AI assistant. Be concise and accurate."},
        {"LRM",        "Think step by step. Break problems down, explore options, check logic, "
                       "then give a FINAL ANSWER."},
        {"SLM",        "Be brief and accurate."},
        {"Agentic",    "You are an action AI. Plan steps, then execute. List tools you would use."},
        {"OpenSource", "You are a powerful open-weight AI assistant. Be thorough."},
        {"Specialized","You are a domain expert. Be precise and structured."},
        {"MoE",        "You are a multi-expert AI. Route this to the most relevant expertise."},
        {"VLM",        "You are a vision-language AI. Describe what you see and analyze."},
        {nullptr, nullptr}
    };

    const char* sys_prompt = "You are a helpful AI assistant.";
    for (int i = 0; AGENTS[i].type; ++i) {
        if (agent_type == AGENTS[i].type) {
            sys_prompt = AGENTS[i].sys;
            break;
        }
    }

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  Agent: %-33s║\n", agent_type.c_str());
    printf("╚══════════════════════════════════════════╝\n");
    printf("Prompt: %s\n\nResponse:\n", user_prompt.c_str());

    auto result = run_completion(model, ctx, sys_prompt, user_prompt);

    printf("\n\n--- Stats ---\n");
    printf("Tokens: %d | Time: %.1f ms | Speed: %.1f tok/s\n\n",
           result.tokens_generated, result.elapsed_ms, result.tokens_per_sec);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <model.gguf> <prompt> [agent_type]\n"
            "  agent_type: LLM | LRM | SLM | Agentic | OpenSource | Specialized | MoE | VLM\n"
            "  Example: %s models/llama-3.1-8b.Q4_K_M.gguf \"Explain AI agents\" LRM\n",
            argv[0], argv[0]);
        return 1;
    }

    const char* model_path  = argv[1];
    const char* user_prompt = argv[2];
    std::string agent_type  = (argc >= 4) ? argv[3] : "LLM";

    HardwareInfo hw = detect_hw();

    // ── Load model ─────────────────────────────────────────────────────────────
    llama_model_params m_params = llama_model_default_params();
    m_params.n_gpu_layers = hw.gpu_layers;

    fprintf(stderr, "[INFO] Loading model: %s\n", model_path);
    llama_model* model = llama_model_load_from_file(model_path, m_params);
    if (!model) {
        fprintf(stderr, "[ERROR] Failed to load model. Check path: %s\n", model_path);
        return 1;
    }

    // ── Create context ────────────────────────────────────────────────────────
    llama_context_params c_params = llama_context_default_params();
    c_params.n_ctx    = 4096;
    c_params.n_batch  = 512;
    c_params.n_threads = 8;

    llama_context* ctx = llama_init_from_model(model, c_params);
    if (!ctx) {
        fprintf(stderr, "[ERROR] Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // ── Run agent ─────────────────────────────────────────────────────────────
    run_agent(model, ctx, agent_type, user_prompt);

    // ── If no agent specified, run ALL 8 types ────────────────────────────────
    if (argc < 4) {
        std::vector<std::string> all_types = {
            "LLM", "MoE", "LRM", "SLM", "Agentic", "OpenSource", "Specialized", "VLM"
        };
        std::string benchmark_prompt = "Explain the concept of attention mechanism in transformers in 2 sentences.";
        printf("\n\n🏟️  Running all 8 agent types on benchmark prompt...\n");
        for (const auto& type : all_types) {
            run_agent(model, ctx, type, benchmark_prompt);
            llama_kv_cache_clear(ctx);    // reset KV cache between agents
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
